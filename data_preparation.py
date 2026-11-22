"""Wyszukiwanie plików Parquet i wczytywanie danych przejazdów."""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from config import MIN_MEAN, PARQUET_FILE_TEMPLATE, RIDE_COLUMNS, RAIN_OPTIONS
from prepare_weather_data import prepare_weather_rain_data


def find_parquet_files(data_dir):
    """Zwróć uporządkowane pliki z danymi przejazdów."""
    return sorted(Path(data_dir).glob(PARQUET_FILE_TEMPLATE.format(year="*")))


def get_available_years(data_dir):
    """Odczytaj dostępne lata z nazw plików Parquet."""
    return sorted(
        int(file.stem.split("_")[-1])
        for file in find_parquet_files(data_dir)
    )


@st.cache_data(show_spinner="Wczytywanie danych...")
def load_rides_data(data_dir, years):
    """Wczytaj wybrane kolumny wyłącznie dla zaznaczonych lat."""
    data_frames = []

    for year in sorted(set(years)):
        file_path = Path(data_dir) / PARQUET_FILE_TEMPLATE.format(year=year)
        ride_data = pd.read_parquet(file_path, columns=RIDE_COLUMNS)
        data_frames.append(ride_data)

    return pd.concat(data_frames, ignore_index=True)

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


def prepare_daily_counts(rides_data, years, month_number=None, day_type="Wszystkie dni", hour_range=None, selected_rain_option="Wszystkie dni"):
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
    if selected_rain_option in RAIN_OPTIONS[1:] and not days.empty:
        weather_data = prepare_weather_rain_data(
            days.min(),
            days.max()
        )
        if selected_rain_option == "Dni deszczowe":
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
