import logging
import os

import requests
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# ============================================================
# GEOCODE / SHOW-ON-MAP TOOL
#
# PROVIDER: OpenStreetMap Nominatim (free, no key required).
#
# Unlike search_nearby_places (which finds MULTIPLE places of a
# CATEGORY around a known lat/long) and get_location (which asks
# the FRONTEND for the user's OWN current location), this tool
# forward-geocodes a single NAMED place the user mentioned (e.g.
# "Vijay Nagar, Indore", "Bargi Dam", "Jabalpur station") into
# real coordinates, so the frontend can drop a single pin/marker
# on a map.
#
# Nominatim's usage policy caps the public instance at 1 request/
# second and requires a descriptive User-Agent — same constraints
# as the Overpass mirrors used in places_tool.py. Override
# NOMINATIM_URL in .env to point at a self-hosted instance for
# heavier traffic.
#
# Never invents coordinates: if Nominatim finds nothing, this
# returns an honest not-found result, not a guessed location.
# ============================================================

NOMINATIM_URL = os.getenv(
    "NOMINATIM_URL", "https://nominatim.openstreetmap.org/search"
).rstrip("/")

NOMINATIM_USER_AGENT = os.getenv(
    "NOMINATIM_USER_AGENT", "rag-chatbot-geocode-tool/1.0"
)


@tool
def find_location_on_map(query: str) -> dict:
    """
    Find the coordinates of a single NAMED place so it can be
    shown as a pin on a map — e.g. "show me Vijay Nagar Indore on
    the map", "where is Bargi Dam", "find Jabalpur station on the
    map".

    Use this ONLY when the user names one specific place they
    want located on a map. Do NOT use this for:
      - the user's own current/live location (use get_location).
      - finding multiple places of a category near a known
        location, like "cafes near me" (use search_nearby_places).
      - distance/travel time between two places (use
        get_distance_bw_2_locations or compare_travel_modes).

    Never guess or invent coordinates. If nothing is found for
    the given query, say so rather than making something up.

    Args:
        query: The place name to locate, e.g. "Vijay Nagar,
            Indore" or "Bargi Dam, Jabalpur". Include a city/area
            for better accuracy if the user gave one.
    """

    logger.info(
        "FIND_LOCATION_ON_MAP CALLED | query=%s",
        query,
    )

    if not query or not query.strip():
        return {
            "success": False,
            "error": "empty_query",
            "message": "No place name was given to search for.",
        }

    headers = {"User-Agent": NOMINATIM_USER_AGENT}

    params = {
        "q": query.strip(),
        "format": "json",
        "limit": 1,
        "addressdetails": 1,
    }

    try:
        response = requests.get(
            NOMINATIM_URL,
            params=params,
            headers=headers,
            timeout=15,
        )
        response.raise_for_status()
        results = response.json()

    except requests.Timeout:
        logger.warning(
            "Nominatim geocoding timed out: query=%s", query
        )
        return {
            "success": False,
            "error": "geocoding_api_timeout",
            "message": "The map search took too long to respond.",
        }

    except requests.RequestException as exc:
        logger.warning("Nominatim geocoding failed: %s", exc)
        return {
            "success": False,
            "error": "geocoding_api_request_failed",
            "message": "Unable to find that location right now.",
        }

    except ValueError:
        logger.warning("Nominatim returned a non-JSON response")
        return {
            "success": False,
            "error": "geocoding_api_invalid_response",
            "message": "Unable to find that location right now.",
        }

    if not results:
        return {
            "success": False,
            "error": "location_not_found",
            "query": query,
            "message": f"No location was found for '{query}'.",
        }

    match = results[0]

    try:
        latitude = float(match.get("lat"))
        longitude = float(match.get("lon"))
    except (TypeError, ValueError):
        return {
            "success": False,
            "error": "geocoding_api_invalid_response",
            "message": "Unable to find that location right now.",
        }

    return {
        "success": True,
        "action": "show_map",
        "query": query,
        "name": match.get("display_name"),
        "latitude": latitude,
        "longitude": longitude,
        "address": match.get("display_name"),
        "provider": "openstreetmap_nominatim",
    }