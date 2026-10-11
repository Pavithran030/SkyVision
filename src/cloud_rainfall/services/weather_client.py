"""Open-Meteo Weather API client for proxy rainfall labels and comparisons."""

from datetime import datetime
from typing import Any, Dict, List, Optional
import requests


class WeatherClient:
    """Client for Open-Meteo public meteorological APIs."""

    def __init__(
        self,
        forecast_url: str = "https://api.open-meteo.com/v1/forecast",
        historical_url: str = "https://archive-api.open-meteo.com/v1/archive",
        timeout: int = 10,
    ):
        self.forecast_url = forecast_url
        self.historical_url = historical_url
        self.timeout = timeout

    def fetch_historical_hourly(
        self,
        latitude: float,
        longitude: float,
        start_date: str,  # YYYY-MM-DD
        end_date: str,    # YYYY-MM-DD
        timezone: str = "auto",
    ) -> Dict[str, Any]:
        """
        Fetch hourly historical precipitation and reanalysis weather variables.
        Used strictly as proxy ground-truth labels for research prototyping.
        """
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": [
                "precipitation",
                "rain",
                "temperature_2m",
                "relative_humidity_2m",
                "surface_pressure",
                "wind_speed_10m",
            ],
            "timezone": timezone,
        }
        try:
            response = requests.get(self.historical_url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return {"success": True, "data": data, "error": None}
        except Exception as e:
            return {"success": False, "data": None, "error": str(e)}

    def fetch_live_weather_and_forecast(
        self,
        latitude: float = 13.0827,
        longitude: float = 80.2707,
        timezone: str = "auto",
    ) -> Dict[str, Any]:
        """
        Fetch current weather conditions and hourly forecast for comparison.
        """
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": [
                "temperature_2m",
                "relative_humidity_2m",
                "surface_pressure",
                "wind_speed_10m",
                "precipitation",
            ],
            "hourly": [
                "precipitation",
                "precipitation_probability",
            ],
            "timezone": timezone,
            "forecast_days": 1,
        }
        try:
            response = requests.get(self.forecast_url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return {"success": True, "data": data, "error": None}
        except Exception as e:
            return {"success": False, "data": None, "error": str(e)}
