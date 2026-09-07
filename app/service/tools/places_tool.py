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
# IMPORTANT TOOL-SELECTION RULE:
#
# This tool is ONLY for physical places / POIs that can reasonably
# be represented in OpenStreetMap.
#
# Examples:
#   - cafes
#   - restaurants
#   - pharmacies
#   - hospitals
#   - hotels
#   - parks
#   - malls
#   - temples
#   - petrol pumps
#   - railway/bus stations
#   - tourist attractions
#
# This tool MUST NOT be used for web/commercial/listing searches
# such as:
#   - property listings
#   - real estate listings
#   - houses/flats for sale or rent
#   - 1BHK/2BHK/3BHK properties
#   - property prices
#   - real estate projects
#   - property dealers/brokers
#   - jobs/job listings
#   - products
#   - shopping
#   - services
#   - marketplace results
#   - latest/current listings
#
# Those requests should use tavily_web_search.
#
# ============================================================

# The main public instance (overpass-api.de) has been returning
# HTTP 406 for many callers regardless of headers used — a known,
# unresolved issue on the Overpass side. To keep this feature
# working, requests try a short list of free public mirrors in
# order and fall back to the next one on failure.
#
# Fair-use / no SLA on any of them — for heavier traffic than a
# demo, self-host Overpass instead.
#
# Override OVERPASS_URL in .env with a comma-separated list to use
# different mirrors (e.g. your own self-hosted instance first).
#
# ============================================================
# HOW IT WORKS:
#
# 1. `query` is a free-text hint like:
#       "cafes"
#       "famous spots"
#       "pharmacy"
#       "petrol pump"
#
#    It is mapped to one or more OSM tag filters via the keyword
#    table below.
#
# 2. If nothing matches, the implementation can perform a bounded
#    broad nearby physical-place search.
#
# 3. Overpass is queried for named nodes/ways of that category
#    within `radius_meters` of the given coordinates.
#
# 4. Results are sorted by straight-line (haversine) distance
#    from the search center.
#
# 5. This distance is ONLY for ranking which place is nearest.
#    If the user wants actual route distance/ETA to a specific
#    place, the LLM should follow up with:
#       get_distance_bw_2_locations
#    or:
#       compare_travel_modes
#
# ============================================================
# NEVER INVENTS PLACES:
#
# If all mirrors fail or Overpass returns nothing, this returns
# an honest empty/error result instead of fabricated place names.
# ============================================================


_DEFAULT_OVERPASS_MIRRORS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]

_overpass_env_override = os.getenv("OVERPASS_URL")

if _overpass_env_override:
    OVERPASS_URLS = [
        url.strip()
        for url in _overpass_env_override.split(",")
        if url.strip()
    ]
else:
    OVERPASS_URLS = _DEFAULT_OVERPASS_MIRRORS


# Overpass mirrors are strict about identifying User-Agents;
# a missing/generic one can cause 403/406 responses.
OVERPASS_USER_AGENT = os.getenv(
    "OVERPASS_USER_AGENT",
    "rag-chatbot-places-tool/1.0",
)


DEFAULT_RADIUS_METERS = 5000
MAX_RADIUS_METERS = 20000
MAX_RESULTS = 10


# ============================================================
# CATEGORY FILTERS
# ============================================================

CATEGORY_FILTERS: dict[str, list[str]] = {
    "attraction": [
        '["tourism"="attraction"]',
        '["historic"]',
        '["tourism"="museum"]',
        '["tourism"="viewpoint"]',
    ],
    "cafe": [
        '["amenity"="cafe"]',
    ],
    "restaurant": [
        '["amenity"="restaurant"]',
    ],
    "food": [
        '["amenity"="restaurant"]',
        '["amenity"="fast_food"]',
    ],
    "pharmacy": [
        '["amenity"="pharmacy"]',
    ],
    "hospital": [
        '["amenity"="hospital"]',
        '["amenity"="clinic"]',
    ],
    "hotel": [
        '["tourism"="hotel"]',
    ],
    "bank": [
        '["amenity"="bank"]',
    ],
    "atm": [
        '["amenity"="atm"]',
    ],
    "park": [
        '["leisure"="park"]',
    ],
    "mall": [
        '["shop"="mall"]',
    ],
    "temple": [
        '["amenity"="place_of_worship"]',
    ],
    # Petrol pump / gas station / fuel.
    "fuel": [
        '["amenity"="fuel"]',
    ],
    # Railway/bus stations.
    "station": [
        '["railway"="station"]',
        '["railway"="halt"]',
        '["amenity"="bus_station"]',
    ],
}


DEFAULT_CATEGORY = "attraction"


# ============================================================
# KEYWORD -> CATEGORY
# ============================================================

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

    # Fuel.
    "petrol pump": "fuel",
    "petrol": "fuel",
    "fuel": "fuel",
    "gas station": "fuel",
    "diesel": "fuel",
    "cng": "fuel",
    "pump": "fuel",

    # Stations.
    "station": "station",
    "railway": "station",
    "train": "station",
    "bus stand": "station",
    "bus station": "station",
}


# Canonical categories surfaced to the LLM through the tool
# description.
CANONICAL_CATEGORIES: list[str] = list(CATEGORY_FILTERS.keys())


# Bounded categories used only for broad/ambiguous physical
# nearby searches.
BROAD_CATEGORY_KEYS: list[str] = [
    "attraction",
    "restaurant",
    "cafe",
    "fuel",
    "pharmacy",
    "mall",
]


# ============================================================
# COORDINATE VALIDATION
# ============================================================

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


# ============================================================
# CATEGORY RESOLUTION
# ============================================================

def _resolve_category(query: str | None) -> str | None:
    """
    Keyword-based fallback category resolution.

    Returns None when no physical-place category matches.

    IMPORTANT:
    Unknown categories are NOT treated as attractions.

    This allows the LLM to choose another tool such as
    tavily_web_search for web/commercial searches.
    """

    if not query:
        return None

    query_lower = query.lower()

    for keyword, category in KEYWORD_TO_CATEGORY.items():
        if keyword in query_lower:
            return category

    return None


# ============================================================
# HAVERSINE DISTANCE
# ============================================================

def _haversine_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    radius_km = 6371.0

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1)
        * math.cos(phi2)
        * math.sin(dlambda / 2) ** 2
    )

    return 2 * radius_km * math.asin(math.sqrt(a))


# ============================================================
# OVERPASS QUERY BUILDERS
# ============================================================

def _build_overpass_query(
    latitude: float,
    longitude: float,
    radius: int,
    filters: list[str],
) -> str:
    clauses = []

    for tag_filter in filters:
        clauses.append(
            f"node{tag_filter}(around:{radius},{latitude},{longitude});"
        )
        clauses.append(
            f"way{tag_filter}(around:{radius},{latitude},{longitude});"
        )

    body = "\n  ".join(clauses)

    return (
        f"[out:json][timeout:20];\n"
        f"(\n"
        f"  {body}\n"
        f");\n"
        f"out center {MAX_RESULTS * 3};"
    )


def _build_broad_query(
    latitude: float,
    longitude: float,
    radius: int,
) -> str:
    """
    Used only when the request is an ambiguous physical nearby
    search such as:

        "what's around me?"
        "what can I find nearby?"

    It unions a bounded set of common physical-place categories.

    IMPORTANT:
    The LLM should NOT use this fallback for property,
    real-estate, product, job, service, marketplace, or other
    web/commercial requests. Those belong to tavily_web_search.
    """

    all_filters: list[str] = []

    for key in BROAD_CATEGORY_KEYS:
        all_filters.extend(CATEGORY_FILTERS[key])

    return _build_overpass_query(
        latitude,
        longitude,
        radius,
        all_filters,
    )


def _build_named_query(
    latitude: float,
    longitude: float,
    radius: int,
    place_name: str,
    filters: list[str] | None,
) -> str:
    """
    Supports searching for a specific named physical business/place.

    Example:
        Starbucks
        McDonald's
        Vijay Nagar Udhyaan

    This is for physical places, not web listings.
    """

    safe_name = (
        place_name
        .replace("\\", "\\\\")
        .replace('"', '\\"')
    )

    name_clause = f'["name"~"{safe_name}",i]'

    if filters:
        tag_filters = [
            f"{tag_filter}{name_clause}"
            for tag_filter in filters
        ]
    else:
        tag_filters = [name_clause]

    clauses = []

    for tag_filter in tag_filters:
        clauses.append(
            f"node{tag_filter}(around:{radius},{latitude},{longitude});"
        )
        clauses.append(
            f"way{tag_filter}(around:{radius},{latitude},{longitude});"
        )

    body = "\n  ".join(clauses)

    return (
        f"[out:json][timeout:20];\n"
        f"(\n"
        f"  {body}\n"
        f");\n"
        f"out center {MAX_RESULTS * 3};"
    )


# ============================================================
# OVERPASS REQUEST
# ============================================================

def _query_overpass(
    overpass_query: str,
) -> tuple[dict | None, dict | None]:
    """
    Query Overpass mirrors in order.

    Returns:
        (data, None) on success
        (None, error) when all mirrors fail
    """

    headers = {
        "User-Agent": OVERPASS_USER_AGENT,
    }

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

            return response.json(), None

        except requests.Timeout:
            logger.warning(
                "Overpass mirror timed out: %s",
                mirror_url,
            )

            last_error = {
                "error": "places_api_timeout",
                "message": (
                    "The places service took too long to respond."
                ),
            }

        except requests.RequestException as exc:
            logger.warning(
                "Overpass mirror failed (%s): %s",
                mirror_url,
                exc,
            )

            last_error = {
                "error": "places_api_request_failed",
                "message": (
                    "Unable to search for nearby places right now."
                ),
            }

        except ValueError:
            logger.warning(
                "Overpass mirror returned non-JSON response: %s",
                mirror_url,
            )

            last_error = {
                "error": "places_api_invalid_response",
                "message": (
                    "Unable to search for nearby places right now."
                ),
            }

    return None, (
        last_error
        or {
            "error": "places_api_request_failed",
            "message": (
                "Unable to search for nearby places right now."
            ),
        }
    )


# ============================================================
# RESULT PARSING
# ============================================================

def _parse_elements(
    elements: list[dict],
    latitude: float,
    longitude: float,
    category_label: str,
) -> list[dict[str, Any]]:
    """
    Parse, deduplicate and rank Overpass results.
    """

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

        distance_km = _haversine_km(
            latitude,
            longitude,
            place_lat,
            place_lon,
        )

        seen_names.add(name)

        results.append(
            {
                "name": name,
                "latitude": place_lat,
                "longitude": place_lon,
                "approx_distance_km": round(
                    distance_km,
                    2,
                ),
                "category": category_label,
            }
        )

    results.sort(
        key=lambda place: place["approx_distance_km"]
    )

    return results[:MAX_RESULTS]


# ============================================================
# SEARCH NEARBY PLACES TOOL
# ============================================================

@tool
def search_nearby_places(
    latitude: float,
    longitude: float,
    query: str | None = None,
    radius_meters: int | None = None,
    category_hint: str | None = None,
    place_name: str | None = None,
) -> dict:
    """
    Search for physical places / points of interest (POIs) near a
    given latitude/longitude using free OpenStreetMap data.

    ============================================================
    USE THIS TOOL FOR PHYSICAL PLACES / POIs
    ============================================================

    Use this tool for places that physically exist at a location,
    especially common POIs such as:

      - cafes
      - coffee shops
      - restaurants
      - food places
      - pharmacies
      - hospitals
      - clinics
      - hotels
      - banks
      - ATMs
      - parks
      - gardens
      - malls
      - temples
      - petrol pumps
      - fuel stations
      - railway stations
      - bus stations
      - tourist attractions
      - museums
      - viewpoints
      - sightseeing spots

    Examples:

      "cafes near me"
          -> search_nearby_places

      "restaurants near me"
          -> search_nearby_places

      "petrol pumps near me"
          -> search_nearby_places

      "parks near me"
          -> search_nearby_places

      "famous places near me"
          -> search_nearby_places

      "find Starbucks near me"
          -> search_nearby_places
             with place_name="Starbucks"

    ============================================================
    DO NOT USE THIS TOOL FOR WEB / COMMERCIAL SEARCHES
    ============================================================

    Do NOT use this tool when the user is asking for information
    that normally comes from websites, listings, marketplaces,
    property portals, shopping websites, job websites, or other
    current web sources.

    Examples:

      - property listings
      - real estate listings
      - properties for sale
      - properties for rent
      - houses for sale
      - houses for rent
      - flats for sale
      - flats for rent
      - apartments for sale
      - apartments for rent
      - 1BHK properties
      - 2BHK properties
      - 3BHK properties
      - real estate projects
      - property dealers
      - property brokers
      - property prices
      - real estate prices
      - latest property listings
      - commercial properties for sale/rent
      - jobs
      - job listings
      - products
      - shopping results
      - services
      - marketplace results
      - online offers
      - current listings
      - current web recommendations

    For these requests, use:

        tavily_web_search

    ============================================================
    VERY IMPORTANT TOOL-SELECTION EXAMPLES
    ============================================================

    "properties near me"
        -> tavily_web_search

    "property for sale near me"
        -> tavily_web_search

    "property for rent near me"
        -> tavily_web_search

    "2BHK for rent near me"
        -> tavily_web_search

    "best properties in Vijay Nagar"
        -> tavily_web_search

    "real estate projects near me"
        -> tavily_web_search

    "property prices in Vijay Nagar"
        -> tavily_web_search

    "latest property listings in Indore"
        -> tavily_web_search

    "houses for sale near me"
        -> tavily_web_search

    "apartments for rent near me"
        -> tavily_web_search

    The words "near me", "nearby", or the presence of known
    latitude/longitude do NOT automatically mean this tool should
    be used.

    First determine WHAT the user is searching for.

    If it is a physical POI such as a cafe, restaurant, park,
    pharmacy, petrol pump, station, etc., use this tool.

    If it is a property, real-estate listing, product, job,
    service, marketplace result, current listing, or other
    web/commercial information, use tavily_web_search.

    ============================================================
    LOCATION REQUIREMENT
    ============================================================

    Call this tool ONLY after a real latitude/longitude is known.

    The coordinates may come from:

      - the user's current location supplied by the frontend
      - a previously known location in the conversation

    Never guess or invent latitude/longitude.

    ============================================================
    SEMANTIC CATEGORY USAGE
    ============================================================

    Prefer passing a normalized category_hint based on the user's
    physical-place intent.

    Valid categories include:

        attraction
        cafe
        restaurant
        food
        pharmacy
        hospital
        hotel
        bank
        atm
        park
        mall
        temple
        fuel
        station

    Examples:

      "I need somewhere to refuel my bike"
          -> category_hint="fuel"

      "I'm hungry, what's around me?"
          -> category_hint="food"

      "show me cafes nearby"
          -> category_hint="cafe"

    Do NOT create a category_hint for web/commercial requests.

    For example:

      "properties near me"
          -> do NOT use category_hint="property"

      "real estate near me"
          -> do NOT use category_hint="real_estate"

    Instead, use:

        tavily_web_search

    ============================================================
    QUERY FALLBACK
    ============================================================

    If category_hint is omitted, the query string is matched
    against the deterministic physical-place keyword table.

    If neither category_hint nor query resolves to a physical-place
    category, a broad nearby physical-place search may be performed.

    However, this broad fallback is intended ONLY for genuinely
    ambiguous physical nearby queries such as:

      "what's around me?"
      "what can I find nearby?"

    It must NOT be used for:

      - property
      - real estate
      - products
      - jobs
      - services
      - marketplaces
      - listings
      - shopping
      - other web/commercial requests

    Those should use tavily_web_search.

    ============================================================
    SPECIFIC PHYSICAL PLACE SEARCH
    ============================================================

    If the user names one specific physical business/place, use
    place_name.

    Example:

      "Find Starbucks near me"

    -> place_name="Starbucks"

    Optionally combine place_name with category_hint for precision.

    ============================================================
    FOLLOW-UPS
    ============================================================

    For physical-place follow-ups such as:

      "another one"
      "which one is closest?"
      "show me something else nearby"

    reuse the same category_hint/place_name from the previous
    physical-place search.

    Use get_conversation_history if the immediate context does
    not make it clear.

    Use the already-known location.

    ============================================================
    DISTANCE BEHAVIOR
    ============================================================

    Distances returned by this tool are straight-line approximate
    distances used only for ranking.

    If the user wants actual route distance or travel time to a
    specific physical place, use:

        get_distance_bw_2_locations

    or:

        compare_travel_modes

    using the place's returned coordinates.

    ============================================================
    PROVIDER
    ============================================================

    This tool searches OpenStreetMap / Overpass.

    It is NOT a general web search tool.

    For property listings, real estate, products, jobs, services,
    marketplace results, current listings, and other web-based
    information, use tavily_web_search.

    Args:
        latitude:
            Latitude of the search center, decimal degrees.

        longitude:
            Longitude of the search center, decimal degrees.

        query:
            Physical-place search hint such as "cafes",
            "restaurants", "pharmacy", "petrol pump",
            "railway station", "famous spots", or
            "tourist attractions".

        radius_meters:
            Search radius in meters.
            Defaults to 5 km and is capped at 20 km.

        category_hint:
            Optional normalized physical-place category.
            Preferred over query when the physical-place intent
            is already understood.

        place_name:
            Optional specific physical business/place name,
            such as "Starbucks".

    IMPORTANT:
        This tool searches physical OpenStreetMap places.

        It must NOT be used as a replacement for
        tavily_web_search.
    """

    logger.info(
        "SEARCH_NEARBY_PLACES CALLED | latitude=%s longitude=%s "
        "query=%s radius_meters=%s category_hint=%s place_name=%s",
        latitude,
        longitude,
        query,
        radius_meters,
        category_hint,
        place_name,
    )

    coord_error = _validate_coordinates(
        latitude,
        longitude,
    )

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

    radius = max(
        100,
        min(radius, MAX_RADIUS_METERS),
    )

    place_name = (
        place_name.strip()
        if place_name
        else None
    )

    # ------------------------------------------------------------
    # RESOLVE CATEGORY
    #
    # 1. category_hint from LLM semantic understanding
    # 2. deterministic keyword fallback
    # 3. None -> broad physical-place fallback
    # ------------------------------------------------------------

    resolved_category: str | None = None

    if (
        category_hint
        and category_hint.strip().lower() in CATEGORY_FILTERS
    ):
        resolved_category = category_hint.strip().lower()
    else:
        resolved_category = _resolve_category(query)

    filters = (
        CATEGORY_FILTERS[resolved_category]
        if resolved_category
        else None
    )

    broad_search = False

    # ------------------------------------------------------------
    # BUILD QUERY
    # ------------------------------------------------------------

    if place_name:
        overpass_query = _build_named_query(
            latitude,
            longitude,
            radius,
            place_name,
            filters,
        )

        category_label = (
            resolved_category
            or "named_place"
        )

    elif resolved_category:
        overpass_query = _build_overpass_query(
            latitude,
            longitude,
            radius,
            filters,
        )

        category_label = resolved_category

    else:
        # This remains the existing broad physical-place fallback.
        #
        # The LLM-facing docstring explicitly tells the model not
        # to reach this path for property/real-estate/web queries.
        overpass_query = _build_broad_query(
            latitude,
            longitude,
            radius,
        )

        category_label = "general"
        broad_search = True

    # ------------------------------------------------------------
    # QUERY OVERPASS
    # ------------------------------------------------------------

    data, error = _query_overpass(
        overpass_query
    )

    if data is None:
        logger.error(
            "All Overpass mirrors failed for places search"
        )

        return {
            "success": False,
            **error,
        }

    results = _parse_elements(
        data.get("elements", []),
        latitude,
        longitude,
        category_label,
    )

    # ------------------------------------------------------------
    # RADIUS ESCALATION
    # ------------------------------------------------------------

    radius_expanded = False

    if (
        not results
        and radius < MAX_RADIUS_METERS
    ):
        expanded_radius = min(
            radius * 2,
            MAX_RADIUS_METERS,
        )

        if expanded_radius > radius:

            if place_name:
                retry_query = _build_named_query(
                    latitude,
                    longitude,
                    expanded_radius,
                    place_name,
                    filters,
                )

            elif resolved_category:
                retry_query = _build_overpass_query(
                    latitude,
                    longitude,
                    expanded_radius,
                    filters,
                )

            else:
                retry_query = _build_broad_query(
                    latitude,
                    longitude,
                    expanded_radius,
                )

            retry_data, _retry_error = _query_overpass(
                retry_query
            )

            if retry_data is not None:
                retry_results = _parse_elements(
                    retry_data.get("elements", []),
                    latitude,
                    longitude,
                    category_label,
                )

                if retry_results:
                    results = retry_results
                    radius = expanded_radius
                    radius_expanded = True

    # ------------------------------------------------------------
    # RESULT NOTE
    # ------------------------------------------------------------

    note = (
        "approx_distance_km is straight-line distance, for ranking "
        "only. For actual route distance and travel time to a "
        "specific place, call get_distance_bw_2_locations or "
        "compare_travel_modes."
    )

    if broad_search:
        note += (
            " No specific physical-place category was identified "
            "for this query, so a broader nearby physical-place "
            "search across common categories was performed."
        )

    # ------------------------------------------------------------
    # FINAL RESULT
    # ------------------------------------------------------------

    return {
        "success": True,
        "latitude": latitude,
        "longitude": longitude,
        "query": query,
        "category_used": category_label,
        "broad_search": broad_search,
        "radius_meters": radius,
        "radius_expanded": radius_expanded,
        "count": len(results),
        "places": results,
        "note": note,
        "provider": "openstreetmap_overpass",
    }