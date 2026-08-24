import logging
import os
from typing import Any

import requests
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


GEOCODING_URL = os.getenv("GEOCODING_URL")
WEATHER_URL = os.getenv("WEATHER_URL")


def _get_location(location: str) -> dict[str, Any]:
    """Convert location name into coordinates."""

    if not GEOCODING_URL:
        raise RuntimeError(
            "GEOCODING_URL is not configured."
        )

    response = requests.get(
        GEOCODING_URL,
        params={
            "name": location,
            "count": 1,
            "language": "en",
            "format": "json",
        },
        timeout=10,
    )

    response.raise_for_status()

    data = response.json()

    results = data.get("results", [])

    if not results:
        raise ValueError(
            f"Location not found: {location}"
        )

    return results[0]


@tool
def get_weather(location: str) -> dict[str, Any]:
    """
    Get the current weather conditions for a specified
    city or location.
    """

    if not location or not location.strip():
        raise ValueError(
            "Weather location cannot be empty."
        )

    if not WEATHER_URL:
        raise RuntimeError(
            "WEATHER_URL is not configured."
        )

    location = location.strip()

    logger.info(
        "WEATHER SEARCH START: location=%s",
        location,
    )

    try:
        location_data = _get_location(location)

        latitude = location_data["latitude"]
        longitude = location_data["longitude"]

        weather_response = requests.get(
            WEATHER_URL,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": (
                    "temperature_2m,"
                    "relative_humidity_2m,"
                    "apparent_temperature,"
                    "is_day,"
                    "precipitation,"
                    "rain,"
                    "weather_code,"
                    "cloud_cover,"
                    "wind_speed_10m,"
                    "wind_direction_10m,"
                ),
                "temperature_unit": "celsius",
                "wind_speed_unit": "kmh",
                "timezone": "auto",
            },
            timeout=10,
        )

        weather_response.raise_for_status()

        data = weather_response.json()

        current = data.get("current")

        if not current:
            raise ValueError(
                "Current weather data was not returned."
            )

        result = {
            "location": location_data.get("name"),
            "country": location_data.get("country"),
            "timezone": location_data.get("timezone"),
            "current": {
                "time": current.get("time"),
                "temperature_celsius": current.get(
                    "temperature_2m"
                ),
                "relative_humidity_percent": current.get(
                    "relative_humidity_2m"
                ),
                "apparent_temperature_celsius": current.get(
                    "apparent_temperature"
                ),
                "precipitation_mm": current.get(
                    "precipitation"
                ),
                "rain_mm": current.get("rain"),
                "weather_code": current.get(
                    "weather_code"
                ),
                "cloud_cover_percent": current.get(
                    "cloud_cover"
                ),
                "wind_speed_kmh": current.get(
                    "wind_speed_10m"
                ),
                "wind_direction_degrees": current.get(
                    "wind_direction_10m"
                ),
                "is_day": current.get("is_day"),
            },
        }

        logger.info(
            "WEATHER SEARCH COMPLETE: location=%s",
            location,
        )

        return result

    except requests.RequestException as exc:
        logger.exception(
            "WEATHER API ERROR: location=%s",
            location,
        )

        raise RuntimeError(
            "Unable to fetch weather information."
        ) from exc

    except Exception:
        logger.exception(
            "WEATHER TOOL ERROR: location=%s",
            location,
        )
        raise