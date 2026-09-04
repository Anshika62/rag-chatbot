import logging
import math
import os
from typing import Any

import requests
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# ============================================================
# SEARCH NEARBY PLACES TOOL
#
# PROVIDER: OpenStreetMap Overpass API (free, no key required).
#
# The main public instance (overpass-api.de) has been returning
# HTTP 406 for many callers regardless of headers used — a known,
# unresolved issue on the Overpass side (see
# github.com/drolbr/Overpass-API/issues/791), not something fixable
# from the client. To keep this feature working, requests try a
# short list of free public mirrors in order and fall back to the
# next one on failure. Fair-use / no SLA on any of them — for
# heavier traffic than a demo, self-host Overpass instead.
#
# Override OVERPASS_URL in .env with a comma-separated list to use
# different mirrors (e.g. your own self-hosted instance first).
#
# HOW IT WORKS:
#   1. `query` (a free-text hint like "cafes", "famous spots",
#      "pharmacy", "petrol pump") is mapped to one or more OSM
#      tag filters via a small keyword table below. If nothing
#      matches, it defaults to "tourist attraction"-style tags
#      (tourism=attraction, historic=*, museum, viewpoint) — this
#      covers "famous spots near me" style queries.
#   2. Overpass is queried for named nodes/ways of that category
#      within `radius_meters` of the given coordinates.
#   3. Results are sorted by straight-line (haversine) distance
#      from the search center — this is ONLY for ranking "which
#      is nearest", not a routed distance. If the user wants an
#      actual route distance/ETA to a specific place, the LLM
#      should follow up with get_distance_bw_2_locations or
#      compare_travel_modes using that place's coordinates.
#
# Never invents places: if all mirrors fail or Overpass returns
# nothing, this returns an honest empty/error result, not
# fabricated place names.
# ============================================================

_DEFAULT_OVERPASS_MIRRORS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]

_overpass_env_override = os.getenv("OVERPASS_URL")

if _overpass_env_override:
    OVERPASS_URLS = [
        url.strip() for url in _overpass_env_override.split(",") if url.strip()
    ]
else:
    OVERPASS_URLS = _DEFAULT_OVERPASS_MIRRORS

# Overpass mirrors are strict about identifying User-Agents; a
# missing/generic one (e.g. bare "python-requests") is a common
# cause of 403/406 responses. Customize the contact bit via
# OVERPASS_USER_AGENT if you have a project URL/email to put there.
OVERPASS_USER_AGENT = os.getenv(
    "OVERPASS_USER_AGENT", "rag-chatbot-places-tool/1.0"
)

DEFAULT_RADIUS_METERS = 5000
MAX_RADIUS_METERS = 20000
MAX_RESULTS = 10

# Category -> list of Overpass tag filter fragments.
CATEGORY_FILTERS: dict[str, list[str]] = {
    "attraction": [
        '["tourism"="attraction"]',
        '["historic"]',
        '["tourism"="museum"]',
        '["tourism"="viewpoint"]',
    ],
    "cafe": ['["amenity"="cafe"]'],
    "restaurant": ['["amenity"="restaurant"]'],
    "food": ['["amenity"="restaurant"]', '["amenity"="fast_food"]'],
    "pharmacy": ['["amenity"="pharmacy"]'],
    "hospital": ['["amenity"="hospital"]', '["amenity"="clinic"]'],
    "hotel": ['["tourism"="hotel"]'],
    "bank": ['["amenity"="bank"]'],
    "atm": ['["amenity"="atm"]'],
    "park": ['["leisure"="park"]'],
    "mall": ['["shop"="mall"]'],
    "temple": ['["amenity"="place_of_worship"]'],
    # Petrol pump / gas station / fuel — OSM tags this as
    # amenity=fuel regardless of fuel type (petrol, diesel, CNG,
    # EV charging is separate: amenity=charging_station).
    "fuel": ['["amenity"="fuel"]'],
    # Railway/bus stations — was previously missing entirely,
    # so "nearby stations" silently fell back to "attraction".
    "station": [
        '["railway"="station"]',
        '["railway"="halt"]',
        '["amenity"="bus_station"]',
    ],
}

DEFAULT_CATEGORY = "attraction"

# Free-text keyword -> category. Substring match, checked in order.
KEYWORD_TO_CATEGORY: dict[str, str] = {
    "famous": "attraction",
    "attraction": "attraction",
    "sightseeing": "attraction",
    "tourist": "attraction",
    "monument": "attraction",
    "museum": "attraction",
    "viewpoint": "attraction",
    "spot": "attraction",
    "cafe": "cafe",
    "coffee": "cafe",
    "restaurant": "restaurant",
    "food": "food",
    "eat": "food",
    "pharmacy": "pharmacy",
    "medicine": "pharmacy",
    "medical": "pharmacy",
    "hospital": "hospital",
    "clinic": "hospital",
    "hotel": "hotel",
    "stay": "hotel",
    "bank": "bank",
    "atm": "atm",
    "park": "park",
    "garden": "park",
    "mall": "mall",
    "shopping": "mall",
    "temple": "temple",
    "mandir": "temple",
    "worship": "temple",
    # fuel / petrol pump keywords
    "petrol pump": "fuel",
    "petrol": "fuel",
    "fuel": "fuel",
    "gas station": "fuel",
    "diesel": "fuel",
    "cng": "fuel",
    "pump": "fuel",
    # station keywords
    "station": "station",
    "railway": "station",
    "train": "station",
    "bus stand": "station",
    "bus station": "station",
}


def _validate_coordinates(latitude, longitude) -> str | None:
    if latitude is None or longitude is None:
        return "Coordinates are missing."
    try:
        lat = float(latitude)
        lon = float(longitude)
    except (TypeError, ValueError):
        return "Coordinates are not valid numbers."
    if not (-90 <= lat <= 90):
        return "Latitude must be between -90 and 90."
    if not (-180 <= lon <= 180):
        return "Longitude must be between -180 and 180."
    return None


def _resolve_category(query: str | None) -> str:
    if not query:
        return DEFAULT_CATEGORY
    query_lower = query.lower()
    for keyword, category in KEYWORD_TO_CATEGORY.items():
        if keyword in query_lower:
            return category
    return DEFAULT_CATEGORY


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * radius_km * math.asin(math.sqrt(a))


def _build_overpass_query(
    latitude: float, longitude: float, radius: int, filters: list[str]
) -> str:
    clauses = []
    for tag_filter in filters:
        clauses.append(f"node{tag_filter}(around:{radius},{latitude},{longitude});")
        clauses.append(f"way{tag_filter}(around:{radius},{latitude},{longitude});")
    body = "\n  ".join(clauses)
    return f"[out:json][timeout:20];\n(\n  {body}\n);\nout center {MAX_RESULTS * 3};"


@tool
def search_nearby_places(
    latitude: float,
    longitude: float,
    query: str | None = None,
    radius_meters: int | None = None,
) -> dict:
    """
    Search for named places (tourist attractions, cafes,
    restaurants, pharmacies, hotels, banks, parks, malls, temples,
    petrol pumps/fuel stations, railway/bus stations, etc.) near a
    given latitude/longitude, using free OpenStreetMap data.

    Call this ONLY after a real latitude/longitude is known —
    either because the user just provided their current location
    in response to a get_location request, or because a location
    was already given earlier in the conversation (e.g. a city the
    user named).

    Never guess or invent latitude/longitude, and never invent
    place names — if this returns an empty list, tell the user
    nothing was found rather than making something up.

    Distances returned here are straight-line (approximate), only
    for ranking which places are nearest. If the user wants an
    actual route distance and travel time to one of these places,
    follow up with get_distance_bw_2_locations or
    compare_travel_modes using that place's coordinates.

    Args:
        latitude: Latitude of the search center, decimal degrees.
        longitude: Longitude of the search center, decimal degrees.
        query: What to search for, e.g. "cafes", "restaurants",
            "pharmacy", "petrol pump", "fuel station", "railway
            station", "famous spots", "tourist attractions". Omit
            or use a generic phrase for "what's interesting around
            here" — defaults to tourist attractions.
        radius_meters: Search radius in meters. Omit to use a
            5 km default. Capped at 20 km.
    """

    logger.info(
        "SEARCH_NEARBY_PLACES CALLED | latitude=%s longitude=%s "
        "query=%s radius_meters=%s",
        latitude,
        longitude,
        query,
        radius_meters,
    )

    coord_error = _validate_coordinates(latitude, longitude)
    if coord_error:
        return {
            "success": False,
            "error": "invalid_coordinates",
            "message": coord_error,
        }

    radius = radius_meters or DEFAULT_RADIUS_METERS
    try:
        radius = int(radius)
    except (TypeError, ValueError):
        radius = DEFAULT_RADIUS_METERS
    radius = max(100, min(radius, MAX_RADIUS_METERS))

    category = _resolve_category(query)
    filters = CATEGORY_FILTERS[category]

    overpass_query = _build_overpass_query(latitude, longitude, radius, filters)
    headers = {"User-Agent": OVERPASS_USER_AGENT}

    data = None
    last_error: dict[str, Any] | None = None

    for mirror_url in OVERPASS_URLS:
        try:
            response = requests.post(
                mirror_url,
                data={"data": overpass_query},
                headers=headers,
                timeout=20,
            )
            response.raise_for_status()
            data = response.json()
            break  # success, stop trying further mirrors

        except requests.Timeout:
            logger.warning("Overpass mirror timed out: %s", mirror_url)
            last_error = {
                "error": "places_api_timeout",
                "message": "The places service took too long to respond.",
            }

        except requests.RequestException as exc:
            logger.warning(
                "Overpass mirror failed (%s): %s", mirror_url, exc
            )
            last_error = {
                "error": "places_api_request_failed",
                "message": "Unable to search for nearby places right now.",
            }

        except ValueError:
            logger.warning(
                "Overpass mirror returned non-JSON response: %s", mirror_url
            )
            last_error = {
                "error": "places_api_invalid_response",
                "message": "Unable to search for nearby places right now.",
            }

    if data is None:
        logger.error("All Overpass mirrors failed for places search")
        return {"success": False, **(last_error or {
            "error": "places_api_request_failed",
            "message": "Unable to search for nearby places right now.",
        })}

    elements = data.get("elements", [])

    seen_names: set[str] = set()
    results: list[dict[str, Any]] = []

    for element in elements:
        tags = element.get("tags", {})
        name = tags.get("name")

        if not name or name in seen_names:
            continue

        if element.get("type") == "node":
            place_lat = element.get("lat")
            place_lon = element.get("lon")
        else:
            center = element.get("center", {})
            place_lat = center.get("lat")
            place_lon = center.get("lon")

        if place_lat is None or place_lon is None:
            continue

        distance_km = _haversine_km(latitude, longitude, place_lat, place_lon)

        seen_names.add(name)
        results.append(
            {
                "name": name,
                "latitude": place_lat,
                "longitude": place_lon,
                "approx_distance_km": round(distance_km, 2),
                "category": category,
            }
        )

    results.sort(key=lambda place: place["approx_distance_km"])
    results = results[:MAX_RESULTS]

    return {
        "success": True,
        "latitude": latitude,
        "longitude": longitude,
        "query": query,
        "category_used": category,
        "radius_meters": radius,
        "count": len(results),
        "places": results,
        "note": (
            "approx_distance_km is straight-line distance, for ranking "
            "only. For actual route distance and travel time to a "
            "specific place, call get_distance_bw_2_locations or "
            "compare_travel_modes."
        ),
        "provider": "openstreetmap_overpass",
    }