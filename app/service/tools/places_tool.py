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
# NEW (semantic Places AI): search_nearby_places now also accepts
# category_hint and place_name, so the LLM can pass a normalized
# category (from its own semantic understanding of the user's
# wording) or a specific business name, instead of this file's
# KEYWORD_TO_CATEGORY table being the only intelligence. That table
# is still used — as the deterministic fallback when category_hint
# is not supplied. If neither resolves to a category, a broader
# multi-category search is performed instead of failing or silently
# defaulting to "attraction".
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
# UNCHANGED — still used as the deterministic fallback layer.
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

# NEW: canonical category names, surfaced to the LLM via the tool
# docstring so it can pass a normalized category_hint.
CANONICAL_CATEGORIES: list[str] = list(CATEGORY_FILTERS.keys())

# NEW: bounded set of categories used only for the broad/ambiguous
# fallback search ("what's around me?" with no resolvable category).
# Not an exhaustive union of every category — keeps the Overpass
# query bounded.
BROAD_CATEGORY_KEYS: list[str] = [
    "attraction",
    "restaurant",
    "cafe",
    "fuel",
    "pharmacy",
    "mall",
]


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


def _resolve_category(query: str | None) -> str | None:
    """
    Keyword-based fallback category resolution.

    CHANGED: previously returned DEFAULT_CATEGORY ("attraction") when
    nothing matched, which silently mis-mapped unmatched queries
    (e.g. "I need somewhere to refuel my bike") onto tourist
    attractions. Now returns None on no match, so the caller can
    fall through to a broad multi-category search instead of
    guessing wrong.
    """
    if not query:
        return None
    query_lower = query.lower()
    for keyword, category in KEYWORD_TO_CATEGORY.items():
        if keyword in query_lower:
            return category
    return None


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


def _build_broad_query(latitude: float, longitude: float, radius: int) -> str:
    """
    NEW. Used only when no category could be resolved (no valid
    category_hint, no keyword match, no place_name) — e.g. "what's
    around me?". Unions a bounded set of common categories instead
    of failing or silently defaulting to "attraction".
    """
    all_filters: list[str] = []
    for key in BROAD_CATEGORY_KEYS:
        all_filters.extend(CATEGORY_FILTERS[key])
    return _build_overpass_query(latitude, longitude, radius, all_filters)


def _build_named_query(
    latitude: float,
    longitude: float,
    radius: int,
    place_name: str,
    filters: list[str] | None,
) -> str:
    """
    NEW. Supports searching for a specific named business/place
    (e.g. "Starbucks"), optionally narrowed by a resolved category's
    tag filters. Overpass QL supports name-tag regex matching
    directly, so no new data source is needed.
    """
    safe_name = place_name.replace("\\", "\\\\").replace('"', '\\"')
    name_clause = f'["name"~"{safe_name}",i]'

    if filters:
        tag_filters = [f"{tag_filter}{name_clause}" for tag_filter in filters]
    else:
        tag_filters = [name_clause]

    clauses = []
    for tag_filter in tag_filters:
        clauses.append(f"node{tag_filter}(around:{radius},{latitude},{longitude});")
        clauses.append(f"way{tag_filter}(around:{radius},{latitude},{longitude});")

    body = "\n  ".join(clauses)
    return f"[out:json][timeout:20];\n(\n  {body}\n);\nout center {MAX_RESULTS * 3};"


def _query_overpass(overpass_query: str) -> tuple[dict | None, dict | None]:
    """
    NEW (refactor only — behavior identical to the previous inline
    mirror-fallback loop inside search_nearby_places; extracted so it
    can be reused for the radius-escalation retry below).
    """
    headers = {"User-Agent": OVERPASS_USER_AGENT}
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

    return None, (
        last_error
        or {
            "error": "places_api_request_failed",
            "message": "Unable to search for nearby places right now.",
        }
    )


def _parse_elements(
    elements: list[dict],
    latitude: float,
    longitude: float,
    category_label: str,
) -> list[dict[str, Any]]:
    """
    NEW (refactor only — identical parsing/dedup logic to before,
    extracted so it can run for both the initial query and the
    radius-escalation retry query).
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

        distance_km = _haversine_km(latitude, longitude, place_lat, place_lon)

        seen_names.add(name)
        results.append(
            {
                "name": name,
                "latitude": place_lat,
                "longitude": place_lon,
                "approx_distance_km": round(distance_km, 2),
                "category": category_label,
            }
        )

    results.sort(key=lambda place: place["approx_distance_km"])
    return results[:MAX_RESULTS]


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

    SEMANTIC USAGE (preferred): interpret the user's natural-language
    request yourself and pass a normalized category_hint — one of:
        attraction, cafe, restaurant, food, pharmacy, hospital,
        hotel, bank, atm, park, mall, temple, fuel, station
    For example "I need somewhere to refuel my bike" ->
    category_hint="fuel"; "I'm hungry, what's around me?" ->
    category_hint="food".

    If category_hint is omitted, the query string is matched against
    a deterministic keyword table as a fallback.

    If NEITHER resolves to a category (e.g. "what's around me?",
    "what can I find nearby?"), do NOT treat this as a failure —
    call with both category_hint and query omitted, and a broader
    multi-category nearby search will be performed automatically.

    SPECIFIC PLACE SEARCH: if the user names one business (e.g.
    "Find Starbucks near me"), pass place_name="Starbucks" (optionally
    together with category_hint for precision).

    FOLLOW-UPS: for "another one", "which one is closest?", "show me
    something else nearby" — reuse the same category_hint/place_name
    as the previous places search in this conversation (use
    get_conversation_history if the immediate context doesn't already
    make it clear), together with the already-known location.

    Distances returned here are straight-line (approximate), only
    for ranking which places are nearest. If the user wants an
    actual route distance and travel time to a specific place,
    follow up with get_distance_bw_2_locations or
    compare_travel_modes using that place's coordinates.

    Args:
        latitude: Latitude of the search center, decimal degrees.
        longitude: Longitude of the search center, decimal degrees.
        query: What to search for, e.g. "cafes", "restaurants",
            "pharmacy", "petrol pump", "fuel station", "railway
            station", "famous spots", "tourist attractions". Used
            only as a fallback keyword match when category_hint is
            not given.
        radius_meters: Search radius in meters. Omit to use a
            5 km default. Capped at 20 km.
        category_hint: Optional normalized category — one of the
            canonical categories listed above. Preferred over query
            when you already understand the user's intent.
        place_name: Optional specific business/place name to search
            for (e.g. "Starbucks"), instead of a generic category.
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

    place_name = place_name.strip() if place_name else None

    # ------------------------------------------------------------
    # RESOLVE CATEGORY
    #   1. category_hint from the LLM's semantic understanding
    #   2. deterministic keyword fallback (existing KEYWORD_TO_CATEGORY)
    #   3. None -> broad multi-category fallback below
    # ------------------------------------------------------------

    resolved_category: str | None = None

    if category_hint and category_hint.strip().lower() in CATEGORY_FILTERS:
        resolved_category = category_hint.strip().lower()
    else:
        resolved_category = _resolve_category(query)

    filters = CATEGORY_FILTERS[resolved_category] if resolved_category else None
    broad_search = False

    if place_name:
        overpass_query = _build_named_query(
            latitude, longitude, radius, place_name, filters
        )
        category_label = resolved_category or "named_place"

    elif resolved_category:
        overpass_query = _build_overpass_query(latitude, longitude, radius, filters)
        category_label = resolved_category

    else:
        overpass_query = _build_broad_query(latitude, longitude, radius)
        category_label = "general"
        broad_search = True

    data, error = _query_overpass(overpass_query)

    if data is None:
        logger.error("All Overpass mirrors failed for places search")
        return {"success": False, **error}

    results = _parse_elements(
        data.get("elements", []), latitude, longitude, category_label
    )

    # ------------------------------------------------------------
    # RADIUS ESCALATION (single step, capped at MAX_RADIUS_METERS)
    # ------------------------------------------------------------

    radius_expanded = False

    if not results and radius < MAX_RADIUS_METERS:
        expanded_radius = min(radius * 2, MAX_RADIUS_METERS)

        if expanded_radius > radius:
            if place_name:
                retry_query = _build_named_query(
                    latitude, longitude, expanded_radius, place_name, filters
                )
            elif resolved_category:
                retry_query = _build_overpass_query(
                    latitude, longitude, expanded_radius, filters
                )
            else:
                retry_query = _build_broad_query(
                    latitude, longitude, expanded_radius
                )

            retry_data, _retry_error = _query_overpass(retry_query)

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

    note = (
        "approx_distance_km is straight-line distance, for ranking "
        "only. For actual route distance and travel time to a "
        "specific place, call get_distance_bw_2_locations or "
        "compare_travel_modes."
    )

    if broad_search:
        note += (
            " No specific category was identified for this query, so a "
            "broader nearby search across common categories was "
            "performed instead of failing."
        )

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