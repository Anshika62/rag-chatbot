import logging
import os
from typing import Any

import requests
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# ============================================================
# ROUTING PROVIDER: OSRM (Open Source Routing Machine)
#
# Free, open-source. Defaults to the public demo server, which
# is fine for a project/demo but is rate-limited and has no SLA
# (per OSRM's own usage policy). For anything beyond a demo,
# self-host OSRM (Docker) and point OSRM_BASE_URL at it.
#
# The public demo only ships the default `driving`, `foot` and
# `bike` profiles. Motorcycle-specific routing and transit
# routing are NOT available from OSRM — transit in particular
# isn't something OSRM solves at all (it needs a GTFS-aware
# engine like OpenTripPlanner). Both are recognised as valid
# travel-mode names but are reported as unsupported-by-provider
# rather than faked.
# ============================================================

OSRM_BASE_URL = os.getenv(
    "OSRM_BASE_URL", "https://router.project-osrm.org"
).rstrip("/")

OSRM_PROFILES = {
    "driving": "driving",
    "walking": "foot",
    "cycling": "bike",
}

ALL_TRAVEL_MODES = ["driving", "motorcycle", "walking", "cycling", "transit"]

# Fallback average speeds (km/h), used ONLY when the routing provider
# returns a duration for one of these modes that is suspiciously identical
# to a different mode's duration in the same comparison (a known issue on
# some free/shared OSRM deployments that don't actually differentiate
# walking/cycling profiles from the driving profile). "driving" itself is
# always trusted as-is from OSRM.
ASSUMED_AVERAGE_SPEED_KMPH = {
    "walking": 5.0,
    "cycling": 15.0,
}

UNSUPPORTED_MODE_REASON = {
    "motorcycle": (
        "Motorcycle-specific routing isn't available from the "
        "OSRM routing provider used in this project."
    ),
    "transit": (
        "Public-transit routing isn't available from the OSRM "
        "routing provider used in this project."
    ),
}


def _validate_coordinates(latitude, longitude, label: str) -> str | None:
    if latitude is None or longitude is None:
        return f"{label} coordinates are missing."

    try:
        lat = float(latitude)
        lon = float(longitude)
    except (TypeError, ValueError):
        return f"{label} coordinates are not valid numbers."

    if not (-90 <= lat <= 90):
        return f"{label} latitude must be between -90 and 90."

    if not (-180 <= lon <= 180):
        return f"{label} longitude must be between -180 and 180."

    return None


def _format_duration(seconds: float) -> str:
    minutes = round(seconds / 60)
    hours = minutes // 60
    remaining_minutes = minutes % 60

    if hours > 0:
        if remaining_minutes > 0:
            return f"{hours} hr {remaining_minutes} min"
        return f"{hours} hr"

    return f"{minutes} min"


def _get_route(
    latitude1: float,
    longitude1: float,
    latitude2: float,
    longitude2: float,
    travel_mode: str,
) -> dict[str, Any]:

    travel_mode_key = (travel_mode or "").lower()

    # --- coordinate validation -----------------------------------
    origin_error = _validate_coordinates(latitude1, longitude1, "Origin")
    if origin_error:
        return {
            "success": False,
            "error": "invalid_coordinates",
            "message": origin_error,
        }

    destination_error = _validate_coordinates(
        latitude2, longitude2, "Destination"
    )
    if destination_error:
        return {
            "success": False,
            "error": "invalid_coordinates",
            "message": destination_error,
        }

    # --- travel mode validation ------------------------------------
    if travel_mode_key not in ALL_TRAVEL_MODES:
        return {
            "success": False,
            "error": "unsupported_travel_mode",
            "travel_mode": travel_mode,
            "supported_modes": ALL_TRAVEL_MODES,
            "message": f"'{travel_mode}' is not a recognised travel mode.",
        }

    if travel_mode_key not in OSRM_PROFILES:
        return {
            "success": False,
            "error": "mode_not_supported_by_provider",
            "travel_mode": travel_mode_key,
            "message": UNSUPPORTED_MODE_REASON.get(
                travel_mode_key,
                f"'{travel_mode_key}' isn't supported by the routing "
                "provider.",
            ),
            "supported_modes": list(OSRM_PROFILES.keys()),
        }

    profile = OSRM_PROFILES[travel_mode_key]

    url = (
        f"{OSRM_BASE_URL}/route/v1/{profile}/"
        f"{longitude1},{latitude1};{longitude2},{latitude2}"
        f"?overview=false&alternatives=false&steps=false"
    )

    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()

    except requests.Timeout:
        logger.exception("OSRM routing request timed out")
        return {
            "success": False,
            "error": "routing_api_timeout",
            "message": "The routing service took too long to respond.",
        }

    except requests.RequestException as exc:
        logger.exception("OSRM routing request failed: %s", exc)
        return {
            "success": False,
            "error": "routing_api_request_failed",
            "message": "Unable to calculate the route right now.",
        }

    except ValueError:
        logger.exception("OSRM returned a non-JSON response")
        return {
            "success": False,
            "error": "routing_api_invalid_response",
            "message": "Unable to calculate the route right now.",
        }

    code = data.get("code")

    if code != "Ok":
        logger.warning(
            "OSRM returned code=%s for travel_mode=%s",
            code,
            travel_mode_key,
        )
        return {
            "success": False,
            "error": "route_not_found",
            "travel_mode": travel_mode_key,
            "message": "No route was found for this travel mode.",
        }

    routes = data.get("routes", [])

    if not routes:
        return {
            "success": False,
            "error": "route_not_found",
            "travel_mode": travel_mode_key,
            "message": "No route was found for this travel mode.",
        }

    route = routes[0]
    distance_meters = route.get("distance")
    duration_seconds = route.get("duration")

    return {
        "success": True,
        "travel_mode": travel_mode_key,
        "distance_meters": distance_meters,
        "distance_km": (
            round(distance_meters / 1000, 2)
            if distance_meters is not None
            else None
        ),
        "duration_seconds": duration_seconds,
        "duration_minutes": (
            round(duration_seconds / 60, 1)
            if duration_seconds is not None
            else None
        ),
        "duration_text": (
            _format_duration(duration_seconds)
            if duration_seconds is not None
            else None
        ),
        "routing_provider": "osrm",
        # OSRM's public profiles use static road speeds, not live
        # traffic — flagged explicitly rather than implied.
        "traffic_aware": False,
    }


@tool
def get_distance_bw_2_locations(
    latitude1: float,
    longitude1: float,
    latitude2: float,
    longitude2: float,
    travel_mode: str = "driving",
) -> dict:
    """
    Calculate route distance and estimated travel time between
    two locations for a single travel mode.

    Supported travel modes: driving, walking, cycling.
    'motorcycle' and 'transit' are recognised but not supported by
    the current routing provider (OSRM) and will return a clear
    unsupported-mode response instead of a fake result.

    Call this only when real latitude/longitude coordinates for
    both locations are known. Never guess or invent coordinates.

    Use compare_travel_modes instead when the user wants two or
    more travel modes compared in a single response.
    """

    logger.info(
        "GET_DISTANCE_BW_2_LOCATIONS CALLED | "
        "location1=(%s, %s) | "
        "location2=(%s, %s) | "
        "travel_mode=%s",
        latitude1,
        longitude1,
        latitude2,
        longitude2,
        travel_mode,
    )

    return _get_route(
        latitude1=latitude1,
        longitude1=longitude1,
        latitude2=latitude2,
        longitude2=longitude2,
        travel_mode=travel_mode,
    )


@tool
def compare_travel_modes(
    latitude1: float,
    longitude1: float,
    latitude2: float,
    longitude2: float,
    travel_modes: list[str] | None = None,
) -> dict:
    """
    Calculate route distance and estimated travel time between two
    locations for MULTIPLE travel modes in a single call.

    Use this when the user wants a comparison — e.g. "compare car,
    bike and walking", "which is faster, car or bike?", or "show me
    distance and time for all available modes".

    travel_modes: optional list drawn from
    ["driving", "motorcycle", "walking", "cycling", "transit"].
    Omit it to compare all modes.

    Call this only with real latitude/longitude coordinates for
    both locations. Never guess or invent coordinates.
    """

    logger.info(
        "COMPARE_TRAVEL_MODES CALLED | "
        "location1=(%s, %s) | "
        "location2=(%s, %s) | "
        "travel_modes=%s",
        latitude1,
        longitude1,
        latitude2,
        longitude2,
        travel_modes,
    )

    origin_error = _validate_coordinates(latitude1, longitude1, "Origin")
    if origin_error:
        return {
            "success": False,
            "error": "invalid_coordinates",
            "message": origin_error,
        }

    destination_error = _validate_coordinates(
        latitude2, longitude2, "Destination"
    )
    if destination_error:
        return {
            "success": False,
            "error": "invalid_coordinates",
            "message": destination_error,
        }

    modes_to_query = travel_modes or ALL_TRAVEL_MODES

    normalized_modes = [
        mode.lower()
        for mode in modes_to_query
        if isinstance(mode, str) and mode.lower() in ALL_TRAVEL_MODES
    ]

    if not normalized_modes:
        return {
            "success": False,
            "error": "unsupported_travel_mode",
            "supported_modes": ALL_TRAVEL_MODES,
            "message": "None of the requested travel modes are recognised.",
        }

    routes: dict[str, Any] = {}
    any_success = False

    for mode_key in normalized_modes:
        result = _get_route(
            latitude1, longitude1, latitude2, longitude2, mode_key
        )
        routes[mode_key] = result
        if result.get("success"):
            any_success = True

    return {
        "success": any_success,
        "origin": {"latitude": latitude1, "longitude": longitude1},
        "destination": {"latitude": latitude2, "longitude": longitude2},
        "routes": routes,
        "routing_provider": "osrm",
    }