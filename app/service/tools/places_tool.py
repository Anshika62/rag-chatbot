import logging

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


# ============================================================
# SEARCH NEARBY PLACES TOOL
#
# STATUS: interface/contract only. No external Places/Maps
# provider (Google Places, Mapbox, etc.) exists anywhere in this
# codebase today, so this tool cannot actually return real
# places yet. Per the Part 4 requirement, this file intentionally
# stops at defining a stable tool contract rather than building
# an external provider integration.
#
# WHEN YOU ADD A REAL PROVIDER:
#   Only the body of search_nearby_places() needs to change to
#   call the provider and return real results. The function
#   signature, docstring, and its registration in
#   conversation_tool.py should NOT need to change, so nothing
#   else in the agent/LLM/SSE flow is affected.
#
# HOW COORDINATES REACH THIS TOOL:
#   The agent (LLM) extracts latitude/longitude out of the
#   conversation itself — the same way get_weather already relies
#   on the LLM to extract a place name from user text. Once the
#   user's location arrives as a message (see location endpoint
#   in the conversation router), the LLM sees it in chat history/
#   the current turn and passes it as this tool's arguments.
# ============================================================


@tool
def search_nearby_places(
    latitude: float,
    longitude: float,
    query: str | None = None,
    radius_meters: int | None = None,
) -> dict:
    """
    Search for places (cafes, restaurants, shops, landmarks, or
    other points of interest) near a given latitude/longitude.

    Call this ONLY after a real latitude/longitude is known —
    either because the user just provided their current location
    in response to a get_location request, or because a location
    was already given earlier in the conversation.

    Never guess or invent latitude/longitude. If you don't have
    real coordinates yet, call get_location first instead.

    Args:
        latitude: Latitude of the search center, decimal degrees.
        longitude: Longitude of the search center, decimal degrees.
        query: What to search for, e.g. "cafes", "restaurants",
            "pharmacy". Omit for a general "what's around here"
            search.
        radius_meters: Search radius in meters. Omit to use a
            sensible default once a provider is configured.
    """

    logger.warning(
        "SEARCH_NEARBY_PLACES CALLED BUT NO PROVIDER CONFIGURED: "
        "latitude=%s longitude=%s query=%s radius_meters=%s",
        latitude,
        longitude,
        query,
        radius_meters,
    )

    return {
        "success": False,
        "error": "places_provider_not_configured",
        "latitude": latitude,
        "longitude": longitude,
        "query": query,
        "message": (
            "A valid location was received, but no Places/Maps "
            "provider is configured in this backend yet. Tell "
            "the user that place search isn't available yet, "
            "without inventing any place results."
        ),
    }