"""Funkcje związane z listami wykluczonych stacji."""

import pandas as pd

from config import EXCLUDED_STATIONS_BY_YEAR


def get_excluded_stations(years):
    """Połącz listy wykluczonych stacji dla zaznaczonych lat."""
    excluded_stations = set()

    for year in years:
        excluded_stations.update(EXCLUDED_STATIONS_BY_YEAR[year])

    return excluded_stations


def get_present_excluded_stations(rides_data, years):
    """Znajdź wykluczone stacje obecne w danych, bez usuwania przejazdów."""
    excluded_stations = get_excluded_stations(years)
    station_ids = pd.to_numeric(rides_data["start_station_id"], errors="coerce")

    return set(station_ids.dropna()) & excluded_stations
