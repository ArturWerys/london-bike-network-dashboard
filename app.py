"""Punkt wejścia aplikacji Streamlit i jej interfejs użytkownika."""

from io import BytesIO

import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator

from config import (
    DATA_DIR,
    DAY_TYPE_OPTIONS,
    MIN_MEAN,
    MIN_VARIANCE,
    MONTH_OPTIONS,
    PAGE_TITLE,
    TIME_WINDOW_OPTIONS,
    HOUR_RANGES
)
from data_loader import get_available_years, load_rides_data
from log_binning_aw import log_binning_aw
from station_filters import get_present_excluded_stations


def filter_low_values(data):
    """Filtrowanie wartości średniej i wariancji poniżej minimalnych progów."""
    mask = (
        (data["mean_count"] >= MIN_MEAN)
        & (data["variance"] >= MIN_VARIANCE)
    )
    return data.loc[mask].copy()


def prepare_daily_counts(rides_data, year, month_number=None, hour_range=None):
    """Obliczanie dziennej liczby przejazdów dla wybranego okresu,
    opcjonalnie w określonym przedziale godzinowym."""

    # Określenie analizowanego okresu
    period_start = pd.Timestamp(
        year=year,
        month=month_number or 1,
        day=1
    )

    period_end = period_start + (
        pd.DateOffset(months=1)
        if month_number is not None
        else pd.DateOffset(years=1)
    )

    # Wszystkie dni kalendarzowe wybranego okresu
    days = pd.date_range(
        period_start,
        period_end,
        freq="D",
        inclusive="left"
    )

    # Konwersja dat
    ride_dates = pd.to_datetime(rides_data["start_date"])

    # Filtrowanie według wybranego okresu
    period_mask = (ride_dates >= period_start) & (ride_dates < period_end)

    # Opcjonalne filtrowanie według godzin
    if hour_range is not None:
        period_mask &= ride_dates.dt.hour.between(
            hour_range[0], hour_range[1]
        )

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
    selected_year,
    selected_month,
    hour_range,
    excluded_high_mean_points=1,
    excluded_low_mean_points=0,
):
    """Wyświetl wykres prawa Taylora z chmurą stacji i dopasowaniem."""

    period_label = f"{selected_month} {selected_year}"

    if hour_range is not None:
        period_label += f", {hour_range[0]} - {hour_range[1]} h"
    
    st.subheader(f"Prawo Taylora - {period_label}")

    mean_log = np.asarray(mean_log)
    variance_log = np.asarray(variance_log)

    # Punkty uwzględnione w dopasowaniu
    n = len(mean_log)
    fit_mask = np.zeros(n, dtype=bool)
    fit_mask[excluded_low_mean_points:n - excluded_high_mean_points] = True

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
        st.pyplot(fig)

        # Eksport PNG
        with BytesIO() as image_buffer:
            fig.savefig(
                image_buffer,
                format="png",
                dpi=300,
                bbox_inches="tight"
            )

            month_number = MONTH_OPTIONS.index(selected_month)
            file_period = (
                f"{selected_year}_{month_number:02d}"
                if month_number else str(selected_year)
            )
            st.download_button(
                "Zapisz wykres jako PNG",
                data=image_buffer.getvalue(),
                file_name=f"prawo_taylora_{file_period}.png",
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

        selected_year = st.pills(
            "Rok:",
            options=available_years,
            selection_mode="single",
            default=available_years[-1],
            required=True,
            key="selected_year",
        )

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

    run_analysis = st.button("Uruchom analizę", type="primary")

    # Wyświetlenie danych i wybranych parametrów po uruchomieniu analizy
    if run_analysis:
        rides_data = load_rides_data(str(DATA_DIR), selected_year, selected_year)

        # Sprawdzenie wykluczonych stacji
        present_excluded_stations = get_present_excluded_stations(
            rides_data, selected_year, selected_year
        )

        rides_data_filtered = rides_data[~pd.to_numeric(rides_data["start_station_id"], errors="coerce").isin(present_excluded_stations)]

        month_number = MONTH_OPTIONS.index(selected_month) or None

        hour_range = HOUR_RANGES[
            TIME_WINDOW_OPTIONS.index(selected_time_window)
        ]

        period_rides, daily_counts = prepare_daily_counts(
            rides_data_filtered, selected_year, month_number, hour_range
        )

        # Średnia i wariancja uwzględniają wszystkie dni wybranego okresu.
        mean_and_variance_counts = pd.DataFrame({
            "mean_count": daily_counts.mean(axis=1),
            "variance": daily_counts.var(axis=1, ddof=0),
        }).reset_index()
        chart_data = filter_low_values(mean_and_variance_counts)

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
                selected_year=selected_year,
                selected_month=selected_month,
                hour_range=hour_range,
                excluded_high_mean_points=excluded_high_mean_points,
                excluded_low_mean_points=excluded_low_mean_points,
            )
        st.write(f"Liczba przejazdów: {len(period_rides):,}")
        st.write(f"Liczba stacji: {period_rides['start_station_id'].nunique()}")

        with st.container(border=True):
            st.markdown("**Parametry analizy**")

            # Zakres danych
            col1, col2, col3, col4 = st.columns(4)
            show_parameter(col1, "Rok", selected_year)
            show_parameter(col2, "Miesiąc", selected_month)
            show_parameter(col3, "Rodzaj dnia", selected_day_type)
            show_parameter(col4, "Godziny", selected_time_window)

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


if __name__ == "__main__":
    main()
