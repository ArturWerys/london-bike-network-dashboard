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
RAIN_OPTIONS = ["Wszystkie dni", "Dni deszczowe", "Dni bezdeszczowe"]
# Końce przedziałów są włączne; początek > koniec oznacza przejście przez północ.
TIME_WINDOWS = [
    ("Pełny dzień", None),
    ("06:00-10:00", (6, 9)),
    ("10:00-14:00", (10, 13)),
    ("15:00-19:00", (15, 18)),
    ("19:00-23:00", (19, 22)),
]

ALTERNATIVE_TIME_WINDOWS = [
    ("Pełny dzień", None),
    ("Morning rush (06:00-08:59)", (6, 8)),
    ("Midday (09:00-13:59)", (9, 13)),
    ("Afternoon Peak (14:00-17:59)", (14, 17)),
    ("Evening (18:00-21:59)", (18, 21)),
    ("Night (22:00-05:59)", (22, 5)),
]
HOUR_OPTIONS = [label for label, _ in TIME_WINDOWS]
HOUR_RANGES = [hours for _, hours in TIME_WINDOWS]
ALTERNATIVE_HOUR_OPTIONS = [label for label, _ in ALTERNATIVE_TIME_WINDOWS]
ALTERNATIVE_HOUR_RANGES = [hours for _, hours in ALTERNATIVE_TIME_WINDOWS]

# Nazwy używane przez aplikację oraz alternatywny zestaw opcji.
TIME_WINDOW_OPTIONS = HOUR_OPTIONS
ALTERNATIVE_TIME_WINDOW_OPTIONS = ALTERNATIVE_HOUR_OPTIONS


def format_hour_range(hour_range):
    """Wspólna etykieta dla interfejsu, wykresów i plików CSV."""
    for label, hours in TIME_WINDOWS + ALTERNATIVE_TIME_WINDOWS:
        if hour_range == hours:
            return label
    return f"{hour_range[0]:02d}:00-{hour_range[1]:02d}:59"


LONDON_LATITUDE = 51.5085
LONDON_LONGITUDE = -0.1257
