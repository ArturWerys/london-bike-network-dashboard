from io import BytesIO

import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator
import openmeteo_requests
import requests_cache
from retry_requests import retry

from config import (
    DATA_DIR,
    DAY_TYPE_OPTIONS,
    MIN_MEAN,
    MONTH_OPTIONS,
    PAGE_TITLE,
    TIME_WINDOW_OPTIONS,
    WEATHER_OPTIONS,
    HOUR_RANGES,
    LONDON_LATITUDE,
    LONDON_LONGITUDE
)
from data_loader import get_available_years, load_rides_data
from log_binning_aw import log_binning_aw
from station_filters import get_present_excluded_stations


def filter_low_values(data, min_mean=MIN_MEAN):
    """Filtrowanie średniej i wariancji na podstawie punktu referencyjnego."""

    mask = (
        (data["mean_count"] >= min_mean)
        & np.isfinite(data["mean_count"])
        & np.isfinite(data["variance"])
    )

    filtered_data = data.loc[mask].copy()

    # Punkt o najmnniejszej dopuszczalnej średniej
    reference_point = filtered_data.sort_values(
        ["mean_count", "variance"]
    ).iloc[0]

    min_variance = reference_point["variance"]

    filtered_data = filtered_data[
        filtered_data["variance"] >= min_variance
    ].copy()

    return filtered_data, min_variance

def prepare_daily_counts(rides_data, years, month_number=None, day_type="Wszystkie dni", hour_range=None, selected_weather_option="Wszystkie dni"):
    """Obliczanie dziennej liczby przejazdów dla wybranego okresu,
    opcjonalnie w określonym przedziale godzinowym."""

    # Kalendarz obejmuje tylko zaznaczone lata, także gdy nie są kolejne.
    years = sorted(set(years))
    days = pd.DatetimeIndex([])

    for year in years:
        period_start = pd.Timestamp(year=year, month=month_number or 1, day=1)
        period_end = period_start + (
            pd.DateOffset(months=1)
            if month_number is not None
            else pd.DateOffset(years=1)
        )
        days = days.append(pd.date_range(
            period_start, period_end, freq="D", inclusive="left"
        ))

    # Konwersja dat
    ride_dates = pd.to_datetime(rides_data["start_date"])

    # Filtrowanie według wybranego okresu
    period_mask = ride_dates.dt.year.isin(years)
    if month_number is not None:
        period_mask &= ride_dates.dt.month == month_number

    # Rodzaj dnia ogranicza zarówno przejazdy, jak i kalendarz statystyk.
    if day_type is not None:
        if day_type == "Dni robocze":
            period_mask &= ride_dates.dt.day_of_week < 5
            days = days[days.day_of_week < 5]
        elif day_type == "Weekendy":
            period_mask &= ride_dates.dt.day_of_week >= 5
            days = days[days.day_of_week >= 5]

    # Opcjonalne filtrowanie według godzin
    if hour_range is not None:
        period_mask &= ride_dates.dt.hour.between(
            hour_range[0], hour_range[1]
        )

    # Pogoda ogranicza przejazdy i dni używane do obliczania statystyk.
    if selected_weather_option in WEATHER_OPTIONS[1:] and not days.empty:
        weather_data = prepare_weather_rain_data(
            days.min(),
            days.max()
        )
        if selected_weather_option == "Dni deszczowe":
            weather_mask = weather_data["rain_sum"] > 0
        else:
            weather_mask = weather_data["rain_sum"] == 0
        weather_days = weather_data.loc[weather_mask, "date"]

        # Brak pomiaru nie jest traktowany jako dzień bezdeszczowy.
        days = days.intersection(pd.DatetimeIndex(weather_days))

        period_mask &= ride_dates.dt.normalize().isin(days)

    # Wybrane przejazdy
    period_rides = rides_data.loc[period_mask].copy()
    period_rides["start_date"] = ride_dates.loc[period_mask].dt.normalize()

    # Dzienna liczba przejazdów
    daily_counts = (
        period_rides.groupby(["start_station_id", "start_date"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=days, fill_value=0)
    )

    return period_rides, daily_counts

def prepare_weather_rain_data(start_date, end_date):
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LONDON_LATITUDE,
        "longitude": LONDON_LONGITUDE,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "daily": "rain_sum",
    }
    responses = openmeteo.weather_api(url, params = params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]

    # Process daily data. The order of variables needs to be the same as requested.
    daily = response.Daily()
    daily_rain_sum = daily.Variables(0).ValuesAsNumpy()

    daily_dataframe = pd.DataFrame({
        "date": pd.date_range(
            start = pd.to_datetime(daily.Time(), unit="s", utc=True),
            end = pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq = pd.Timedelta(seconds=daily.Interval()),
            inclusive = "left",
        ).tz_localize(None).normalize(),
        "rain_sum": daily_rain_sum
    })

    return daily_dataframe


def show_parameter(column, label, value):
    """Wyświetl pojedynczy parametr w podanej kolumnie."""
    with column:
        st.caption(label)
        st.markdown(f"**{value}**")


def show_taylor_chart(
    mean_values,
    variance_values,
    mean_log,
    variance_log,
    selected_years,
    selected_month,
    selected_day_type,
    selected_hour_range,
    selected_weather_option,
    excluded_high_mean_points=1,
    excluded_low_mean_points=0,
):
    """Wyświetl wykres prawa Taylora z chmurą stacji i dopasowaniem."""

    years_label = ", ".join(str(year) for year in selected_years)
    period_label = f"{selected_month} {years_label}"

    if selected_day_type != "Wszystkie dni":
        period_label += f", {selected_day_type}"

    if selected_hour_range is not None:
        period_label += f", {selected_hour_range[0]} - {selected_hour_range[1]} h"
    
    if selected_weather_option != "Wszystkie dni":
        period_label += f", {selected_weather_option.lower()}"

    st.subheader(f"Prawo Taylora - {period_label}")

    mean_log = np.asarray(mean_log)
    variance_log = np.asarray(variance_log)

    # Punkty uwzględnione w dopasowaniu
    n = len(mean_log)
    fit_mask = np.zeros(n, dtype=bool)
    fit_mask[excluded_low_mean_points:max(0, n - excluded_high_mean_points)] = True

    if fit_mask.sum() < 2:
        st.warning("Za mało punktów do dopasowania. Obniż progi lub zmniejsz liczbę pomijanych punktów.")
        return

    excluded_mask = ~fit_mask

    # Dopasowanie prawa Taylora
    x = np.log(mean_log[fit_mask])
    y = np.log(variance_log[fit_mask])

    alpha, log_A = np.polyfit(x, y, 1)
    A = np.exp(log_A)

    # Ocena jakości dopasowania
    y_pred = alpha * x + log_A
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    alpha_str = f"{alpha:.3f}".replace(".", ",")
    r_squared_str = f"{r_squared:.3f}".replace(".", ",")

    # Generowanie dopasowanej funkcji
    x_fit = np.geomspace(
        mean_log[fit_mask].min(),
        mean_log[fit_mask].max(),
        100
    )
    y_fit = A * x_fit**alpha

    # Wykres
    fig, ax = plt.subplots(figsize=(10, 7))

    # Chmura punktów – poszczególne stacje
    ax.scatter(
        mean_values,
        variance_values,
        color="#80C4E9",
        alpha=0.5,
        s=36,
        zorder=1,
        label="Poszczególne stacje"
    )

    # Punkty po binowaniu uwzględnione w dopasowaniu
    ax.scatter(
        mean_log[fit_mask],
        variance_log[fit_mask],
        color="#102A43",
        alpha=0.7,
        marker="D",
        s=70,
        zorder=3,
        label="Punkty po binowaniu logarytmicznym"
    )

    # Punkty pominięte w dopasowaniu
    if excluded_mask.any():
        ax.scatter(
            mean_log[excluded_mask],
            variance_log[excluded_mask],
            marker="D",
            facecolors="white",
            edgecolors="#102A43",
            linewidths=1.8,
            s=85,
            zorder=3,
            label="Punkty pominięte w dopasowaniu"
        )

    # Dopasowana funkcja potęgowa
    ax.plot(
        x_fit,
        y_fit,
        color="#102A43",
        linewidth=2.5,
        zorder=2,
        label=fr"Dopasowanie: $\alpha = {alpha_str}$; $R^2 = {r_squared_str}$"
    )

    # Skale i opisy
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel("Średnia dzienna liczba wypożyczeń")
    ax.set_ylabel("Wariancja")

    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(LogLocator(base=10))
        axis.set_major_formatter(
            FuncFormatter(lambda value, _: f"{value:g}")
        )
        axis.set_minor_locator(NullLocator())

    ax.grid(True, which="major", axis="both", alpha=0.3)
    ax.legend(frameon=False)

    fig.suptitle(
        f"Prawo Taylora - {period_label}",
        fontsize=16
    )

    fig.tight_layout()

    try:
        st.pyplot(fig, width=1000)

        # Eksport PNG
        with BytesIO() as image_buffer:
            fig.savefig(
                image_buffer,
                format="png",
                dpi=300,
                bbox_inches="tight"
            )

            month_number = MONTH_OPTIONS.index(selected_month)
            file_years = "_".join(str(year) for year in selected_years)
            file_period = (
                f"{file_years}_{month_number:02d}"
                if month_number else file_years
            )
            weather_suffix = {
                "Wszystkie dni": "",
                "Dni deszczowe": "_deszczowe",
                "Dni bezdeszczowe": "_bezdeszczowe",
            }[selected_weather_option]
            st.download_button(
                "Zapisz wykres jako PNG",
                data=image_buffer.getvalue(),
                file_name=f"prawo_taylora_{file_period}{weather_suffix}.png",
                mime="image/png",
                on_click="ignore",
            )
    finally:
        plt.close(fig)


def main():
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    st.title(PAGE_TITLE)

    available_years = get_available_years(DATA_DIR)
    if not available_years:
        st.warning("Brak dostępnych danych do analizy.")
        return

    # Podział strony na dwie kolumny o równych szerokościach
    col_data_filters, col_fit_filters = st.columns([1, 1], gap="large")

    # Zakres danych
    with col_data_filters:
        st.subheader("Zakres danych")

        if "selected_years" not in st.session_state:
            st.session_state["selected_years"] = [available_years[-1]]

        selected_years = sorted(st.pills(
            "Lata:",
            options=available_years,
            selection_mode="multi",
            key="selected_years",
        ))
        if not selected_years:
            st.info("Wybierz przynajmniej jeden rok, aby uruchomić analizę.")

        selected_month = st.pills(
            "Miesiąc:",
            options=MONTH_OPTIONS,
            selection_mode="single",
            default=MONTH_OPTIONS[0],
            required=True,
            key="selected_month",
        )
        selected_day_type = st.selectbox("Rodzaj dnia:", DAY_TYPE_OPTIONS)
        selected_time_window = st.selectbox("Godziny:", TIME_WINDOW_OPTIONS)

        selected_weather_option = st.radio(
            "Warunki pogodowe:",
            options=WEATHER_OPTIONS,
            index=0,
            horizontal=True,
        )

    # Parametry dopasowania
    with col_fit_filters:
        st.subheader("Parametry dopasowania")

        binning_factor = st.number_input(
            "Współczynnik binowania a:",
            min_value=1.05,
            max_value=3.0,
            value=1.4,
            step=0.05,
        )

        excluded_high_mean_points = st.number_input(
            "Liczba pomijanych w dopasowaniu punktów o największej średniej:",
            min_value=0,
            max_value=10,
            value=1,
            step=1,
        )

        excluded_low_mean_points = st.number_input(
            "Liczba pomijanych w dopasowaniu punktów o najmniejszej średniej:",
            min_value=0,
            max_value=10,
            value=0,
            step=1,
        )
        st.subheader("Filtracja danych")

        min_mean = st.number_input(
            "Minimalna średnia:",
            min_value=0.0,
            value=MIN_MEAN,
            step=0.01,
            format="%.2f",
            key="min_mean",
            help="Minimalna wariancja automatycznie przyjmuje tę samą wartość.",
        )

    run_analysis = st.button(
        "Uruchom analizę", type="primary", disabled=not selected_years
    )

    # Wyświetlenie danych i wybranych parametrów po uruchomieniu analizy
    if run_analysis:
        rides_data = load_rides_data(str(DATA_DIR), tuple(selected_years))

        # Sprawdzenie wykluczonych stacji
        present_excluded_stations = get_present_excluded_stations(
            rides_data, selected_years
        )

        rides_data_filtered = rides_data[~pd.to_numeric(rides_data["start_station_id"], errors="coerce").isin(present_excluded_stations)]

        month_number = MONTH_OPTIONS.index(selected_month) or None

        hour_range = HOUR_RANGES[
            TIME_WINDOW_OPTIONS.index(selected_time_window)
        ]

        period_rides, daily_counts = prepare_daily_counts(
            rides_data_filtered,
            selected_years,
            month_number,
            day_type=selected_day_type,
            hour_range=hour_range,
            selected_weather_option=selected_weather_option
        )

        # Średnia i wariancja uwzględniają wszystkie dni wybranego okresu.
        mean_and_variance_counts = pd.DataFrame({
            "mean_count": daily_counts.mean(axis=1),
            "variance": daily_counts.var(axis=1, ddof=0),
        }).reset_index()
        chart_data, min_variance = filter_low_values(
            mean_and_variance_counts, min_mean=min_mean
        )

        if chart_data.empty:
            st.warning("Brak danych spełniających progi analizy w wybranym okresie.")
        else:
            mean_log, variance_log = log_binning_aw(
                chart_data["mean_count"],
                chart_data["variance"],
                binning_factor
            )

            show_taylor_chart(
                mean_values=chart_data["mean_count"],
                variance_values=chart_data["variance"],
                mean_log=mean_log,
                variance_log=variance_log,
                selected_years=selected_years,
                selected_month=selected_month,
                selected_day_type=selected_day_type,
                selected_hour_range=hour_range,
                selected_weather_option=selected_weather_option,
                excluded_high_mean_points=excluded_high_mean_points,
                excluded_low_mean_points=excluded_low_mean_points,
            )
        st.write(f"Liczba przejazdów: {len(period_rides):,}")
        st.write(f"Liczba stacji: {period_rides['start_station_id'].nunique()}")
        st.write(f"Liczba analizowanych dni: {len(daily_counts.columns)}")

        with st.container(border=True):
            st.markdown("**Parametry analizy**")

            # Zakres danych
            col1, col2, col3, col4, col5 = st.columns(5)
            show_parameter(col1, "Lata", ", ".join(str(year) for year in selected_years))
            show_parameter(col2, "Miesiąc", selected_month)
            show_parameter(col3, "Rodzaj dnia", selected_day_type)
            show_parameter(col4, "Godziny", selected_time_window)
            show_parameter(col5, "Pogoda", selected_weather_option)

            st.divider()

            # Parametry dopasowania
            col1, col2, col3 = st.columns(3)
            show_parameter(col1, "Współczynnik a", f"{binning_factor:.2f}")
            show_parameter(
                col2,
                "Pominięte punkty o największej wartości średniej",
                excluded_high_mean_points,
            )
            show_parameter(
                col3,
                "Pominięte punkty o najmniejszej wartości średniej",
                excluded_low_mean_points,
            )

            col1, col2 = st.columns(2)
            show_parameter(col1, "Minimalna średnia", f"{min_mean:g}")
            show_parameter(col2, "Minimalna wariancja ", f"{min_variance:g}")

    st.divider()

    st.markdown(
        """
        <div style="text-align: center; color: #888; font-size: 15px;">
            Opracowanie: Artur Werys<br>
            Politechnika Warszawska · Wydział Fizyki · 2026
        </div>
        """,
        unsafe_allow_html=True
    )
if __name__ == "__main__":
    main()
