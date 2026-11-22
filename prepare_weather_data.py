import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

from config import LONDON_LATITUDE, LONDON_LONGITUDE


def prepare_weather_rain_data(start_date, end_date):
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LONDON_LATITUDE,
        "longitude": LONDON_LONGITUDE,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "daily": "rain_sum",
    }
    responses = openmeteo.weather_api(url, params = params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]

    # Process daily data. The order of variables needs to be the same as requested.
    daily = response.Daily()
    daily_rain_sum = daily.Variables(0).ValuesAsNumpy()

    daily_dataframe = pd.DataFrame({
        "date": pd.date_range(
            start = pd.to_datetime(daily.Time(), unit="s", utc=True),
            end = pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq = pd.Timedelta(seconds=daily.Interval()),
            inclusive = "left",
        ).tz_localize(None).normalize(),
        "rain_sum": daily_rain_sum
    })

    return daily_dataframe
