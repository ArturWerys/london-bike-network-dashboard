import numpy as np
import pandas as pd
import csv
from pathlib import Path

from config import DATA_DIR, MONTH_OPTIONS, HOUR_RANGES, format_hour_range
from data_preparation import (
    filter_low_values,
    load_rides_data,
    prepare_daily_counts,
)
from station_filters import get_present_excluded_stations

def calculate_alfa(year_type, month_type, day_type, hour_range, rain_option, min_mean, filename):
    
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

    # print(f"Liczba przejazdów: {len(period_rides):,}")
    # print(f"Liczba stacji: {period_rides['start_station_id'].nunique()}")
    # print(f"Liczba analizowanych dni: {len(daily_counts.columns)}")

    # Średnia i wariancja uwzględniają wszystkie dni wybranego okresu.
    mean_and_variance_counts = pd.DataFrame({
        "mean_count": daily_counts.mean(axis=1),
        "variance": daily_counts.var(axis=1, ddof=0),
    }).reset_index()

    # Te same progi średniej i wariancji co w aplikacji przed wykresem.
    eligible = (
        (mean_and_variance_counts["mean_count"] >= min_mean)
        & np.isfinite(mean_and_variance_counts["mean_count"])
        & np.isfinite(mean_and_variance_counts["variance"])
    )
    if not eligible.any():
        raise ValueError("Brak punktów spełniających próg minimalnej średniej.")
    mean_and_variance_counts, _ = filter_low_values(
        mean_and_variance_counts, min_mean=min_mean
    )

    fit_data = mean_and_variance_counts.loc[
        np.isfinite(mean_and_variance_counts["mean_count"])
        & np.isfinite(mean_and_variance_counts["variance"])
    ]
    if len(fit_data) < 2 or fit_data["mean_count"].nunique() < 2:
        raise ValueError("Za mało punktów o różnych średnich do obliczenia alfa.")

    # Dopasowanie prawa Taylora
    x = np.log(fit_data["mean_count"])
    y = np.log(fit_data["variance"])

    alpha, _ = np.polyfit(x, y, 1)

    # Zapisanie wyników do pliku CSV

    plik_csv = Path(__file__).resolve().parent / filename
    zapisz_naglowek = not plik_csv.exists() or plik_csv.stat().st_size == 0

    with plik_csv.open("a", newline="", encoding="utf-8-sig") as plik:
        writer = csv.writer(plik, delimiter=";")

        if zapisz_naglowek:
            writer.writerow([
                "Lata", "Miesiąc", "Dzień", "Godziny",
                "Deszcz", "Minimalna średnia", "Alfa",
            ])

        writer.writerow([
            ", ".join(map(str, year_type)),
            month_type,
            day_type,
            format_hour_range(hour_range),
            rain_option,
            min_mean,
            float(alpha),
        ])

    return float(alpha)


if __name__ == "__main__":

    # Analiza przedziałów godzinowych dla dni roboczych 
    for year in [2023, 2024, 2025]:
        for month in MONTH_OPTIONS:
            for hour_range in HOUR_RANGES[1:]:
                alfa_workday = calculate_alfa(
                    year_type=[year],
                    month_type=month,
                    day_type="Workday",
                    hour_range=hour_range,
                    rain_option="Wszystkie dni",
                    min_mean=0,
                    filename="2023_2024_2025_przedzialy_godzinowe_dni_robocze.csv"
                )
