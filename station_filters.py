"""Funkcje związane z listami wykluczonych stacji."""

import pandas as pd

from config import EXCLUDED_STATIONS_BY_YEAR


def get_excluded_stations(start_year, end_year):
    """Połącz listy wykluczonych stacji dla wybranego zakresu lat."""
    excluded_stations = set()

    for year in range(start_year, end_year + 1):
        excluded_stations.update(EXCLUDED_STATIONS_BY_YEAR[year])

    return excluded_stations


def get_present_excluded_stations(rides_data, start_year, end_year):
    """Znajdź wykluczone stacje obecne w danych, bez usuwania przejazdów."""
    excluded_stations = get_excluded_stations(start_year, end_year)
    station_ids = pd.to_numeric(rides_data["start_station_id"], errors="coerce")

    return set(station_ids.dropna()) & excluded_stations
