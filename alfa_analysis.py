import numpy as np
import pandas as pd

from config import DATA_DIR, MONTH_OPTIONS
from data_preparation import (
    load_rides_data,
    prepare_daily_counts,
)
from station_filters import get_present_excluded_stations

def calculate_alfa(year_type, month_type, day_type, hour_range, rain_option):
    
    month_number = MONTH_OPTIONS.index(month_type) or None

    # Wczytanie danych dla wybranych lat
    rides_data = load_rides_data(DATA_DIR, year_type)

    # Sprawdzenie wykluczonych stacji
    present_excluded_stations = get_present_excluded_stations(
        rides_data, year_type
    )

    rides_data_filtered = rides_data[~pd.to_numeric(rides_data["start_station_id"], errors="coerce").isin(present_excluded_stations)]

    period_rides, daily_counts = prepare_daily_counts(
        rides_data_filtered,
        year_type,
        month_number,
        day_type,
        hour_range,
        rain_option
    )

    print(f"Liczba przejazdów: {len(period_rides):,}")
    print(f"Liczba stacji: {period_rides['start_station_id'].nunique()}")
    print(f"Liczba analizowanych dni: {len(daily_counts.columns)}")

    # Średnia i wariancja uwzględniają wszystkie dni wybranego okresu.
    mean_and_variance_counts = pd.DataFrame({
        "mean_count": daily_counts.mean(axis=1),
        "variance": daily_counts.var(axis=1, ddof=0),
    }).reset_index()

    # Logarytm wymaga dodatnich, skończonych wartości.
    fit_data = mean_and_variance_counts.loc[
        (mean_and_variance_counts["mean_count"] > 0)
        & (mean_and_variance_counts["variance"] > 0)
        & np.isfinite(mean_and_variance_counts["mean_count"])
        & np.isfinite(mean_and_variance_counts["variance"])
    ]
    if len(fit_data) < 2 or fit_data["mean_count"].nunique() < 2:
        raise ValueError("Za mało punktów o różnych średnich do obliczenia alfa.")

    # Dopasowanie prawa Taylora
    x = np.log(fit_data["mean_count"])
    y = np.log(fit_data["variance"])

    alpha, _ = np.polyfit(x, y, 1)
    
    return float(alpha)

if __name__ == "__main__":
    alfa = calculate_alfa(
        year_type=[2023],
        month_type="Cały rok",
        day_type="Wszystkie dni",
        hour_range=None,
        rain_option="Wszystkie dni",
    )
    print(f"Alfa: {alfa:.6f}")
