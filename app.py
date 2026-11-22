import pandas as pd
import streamlit as st

from config import (
    DATA_DIR,
    DAY_TYPE_OPTIONS,
    HOUR_RANGES,
    MIN_MEAN,
    MONTH_OPTIONS,
    PAGE_TITLE,
    TIME_WINDOW_OPTIONS,
    RAIN_OPTIONS,
)
from data_preparation import (
    filter_low_values,
    get_available_years,
    load_rides_data,
    prepare_daily_counts,
)
from log_binning_aw import log_binning_aw
from station_filters import get_present_excluded_stations
from taylors_law_chart import show_taylor_chart


def show_parameter(column, label, value):
    """Wyświetl pojedynczy parametr w podanej kolumnie."""
    with column:
        st.caption(label)
        st.markdown(f"**{value}**")


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

        selected_rain_option = st.radio(
            "Warunki pogodowe:",
            options=RAIN_OPTIONS,
            index=0,
            horizontal=True,
        )

    # Parametry dopasowania
    with col_fit_filters:
        st.subheader("Rodzaj dopasowania")

        selected_type = st.segmented_control(
            "Rodzaj dopasowania:",
            options=["Z binowaniem", "Bez binowania"],
            selection_mode="single",
            default="Z binowaniem",
            required=True,
            label_visibility="collapsed",
            key="selected_type",
        )

        excluded_high_mean_points = 0
        excluded_low_mean_points = 0

        if selected_type == "Z binowaniem":

            st.subheader("Parametry binowania")

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
            

        else:
            st.subheader("Parametry dopasowania bez binowania")

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
            selected_rain_option=selected_rain_option
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

        if selected_type == "Z binowaniem" and not chart_data.empty:
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
                selected_rain_option=selected_rain_option,
                excluded_high_mean_points=excluded_high_mean_points,
                excluded_low_mean_points=excluded_low_mean_points,
                binned=True
            )
        st.write(f"Liczba przejazdów: {len(period_rides):,}")
        st.write(f"Liczba stacji: {period_rides['start_station_id'].nunique()}")
        st.write(f"Liczba analizowanych dni: {len(daily_counts.columns)}")

        if selected_type == "Bez binowania" and not chart_data.empty:
            show_taylor_chart(
                mean_values=chart_data["mean_count"],
                variance_values=chart_data["variance"],
                selected_years=selected_years,
                selected_month=selected_month,
                selected_day_type=selected_day_type,
                selected_hour_range=hour_range,
                selected_weather_option=selected_weather_option,
                excluded_high_mean_points=excluded_high_mean_points,
                excluded_low_mean_points=excluded_low_mean_points,
                binned=False
            )

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
            if selected_type == "Z binowaniem":
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
