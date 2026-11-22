from pathlib import Path
import sys

import pandas as pd

# Umożliwia bezpośrednie uruchomienie skryptu z tego podkatalogu.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    RIDE_COLUMNS,
    PROJECT_DIR,
    RAW_PARQUET_FILE_TEMPLATE,
    PARQUET_FILE_TEMPLATE,
)

DATA_DIR = PROJECT_DIR / "raw_data"
OUTPUT_DIR = PROJECT_DIR / "processed_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for year in (2023, 2024, 2025):
    filename = RAW_PARQUET_FILE_TEMPLATE.format(year=year)
    source_path = DATA_DIR / filename
    output_path = OUTPUT_DIR / PARQUET_FILE_TEMPLATE.format(year=year)

    rides = pd.read_parquet(source_path, columns=RIDE_COLUMNS)
    rides.to_parquet(output_path, index=False)
    del rides

    original_mb = source_path.stat().st_size / 1_000_000
    processed_mb = output_path.stat().st_size / 1_000_000
    print(f"{year}: {original_mb:.2f} MB -> {processed_mb:.2f} MB (mniej o {original_mb - processed_mb:.2f} MB)")
