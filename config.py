"""Ścieżki do danych i stałe konfiguracyjne aplikacji."""

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "processed_data"

RAW_PARQUET_FILE_TEMPLATE = "rides_data_without_discontinuities_{year}.parquet"
PARQUET_FILE_TEMPLATE = "rides_data_to_analyze_{year}.parquet"
RIDE_COLUMNS = [
    "start_station_id",
    "start_station_name",
    "start_date",
    "start_day_of_week",
]

# Domyślny wspólny próg średniej i wariancji, edytowalny w interfejsie.
MIN_MEAN = 0.1

EXCLUDED_STATIONS_BY_YEAR = {
    2023: {1197, 200081, 1057, 300071, 200227},  # to jest prawdziwe
    2024: {200032, 22181},  # to jest prawdziwe
    2025: {300060, 300074, 300099},  # to jest prawdziwe

}

PAGE_TITLE = "Analiza sieci rowerów miejskich Londynu"
MONTH_OPTIONS = [
    "Cały rok", "Styczeń", "Luty", "Marzec", "Kwiecień",
    "Maj", "Czerwiec", "Lipiec", "Sierpień", "Wrzesień",
    "Październik", "Listopad", "Grudzień",
]
DAY_TYPE_OPTIONS = ["Wszystkie dni", "Dni robocze", "Weekendy"]
WEATHER_OPTIONS = ["Wszystkie dni", "Dni deszczowe", "Dni bezdeszczowe"]
TIME_WINDOW_OPTIONS = [
    "Pełny dzień", "06:00-9:00", "10:00-13:00", "15:00-18:00", "19:00-22:00",
]
HOUR_RANGES = [
    None,
    (6, 9),
    (10, 13),
    (15, 18),
    (19, 22)
]

LONDON_LATITUDE = 51.5085
LONDON_LONGITUDE = -0.1257
