"""Wyszukiwanie plików Parquet i wczytywanie danych przejazdów."""

from pathlib import Path

import pandas as pd
import streamlit as st

from config import PARQUET_FILE_TEMPLATE, RIDE_COLUMNS


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
def load_rides_data(data_dir, start_year, end_year):
    """Wczytaj wybrane kolumny dla zakresu lat, włącznie z oboma końcami."""
    data_frames = []

    for year in range(start_year, end_year + 1):
        file_path = Path(data_dir) / PARQUET_FILE_TEMPLATE.format(year=year)
        ride_data = pd.read_parquet(file_path, columns=RIDE_COLUMNS)
        data_frames.append(ride_data)

    return pd.concat(data_frames, ignore_index=True)
