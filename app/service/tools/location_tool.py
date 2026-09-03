import logging

from langchain_core.tools import tool


logger = logging.getLogger(__name__)


# ============================================================
# GET LOCATION TOOL
#
# Unlike get_weather (which geocodes a NAMED place the user
# already mentioned), this tool is for when the user's OWN
# current location is required and has not been provided
# anywhere in the conversation.
#
# This tool does NOT detect or return real coordinates. It has
# no way to know where the user physically is. Calling it is a
# SIGNAL to the application that a location-selection UI should
# be shown to the user on the frontend.
#
# Flow:
#   1. Agent decides the user's location is required.
#   2. Agent calls get_location.
#   3. This function returns a small marker result (below).
#   4. llm_service._execute_tool_calls() detects that marker and
#      surfaces it to generate_answer_stream(), which short-
#      circuits the normal "ask the LLM to write a final answer"
#      step so it can never fabricate a location.
#   5. rag_service.query_documents_stream() turns that into a
#      dedicated "location_request" SSE event for the frontend.
#   6. The frontend shows a location picker and later sends the
#      chosen latitude/longitude/address back as a normal user
#      message in a follow-up turn.
# ============================================================


@tool
def get_location() -> dict:
    """
    Request the user's current location.

    Call this ONLY when answering the current question actually
    requires knowing where the user is right now (for example:
    "restaurants near me", "what's the weather here", "nearest
    branch to my location") AND the location has not already been
    given in the current question or the conversation history.

    Do NOT call this if the user already named a place (use
    get_weather or search_knowledge_base as appropriate instead).

    This tool does not return real coordinates. It triggers a
    location-selection UI on the user's device. The actual
    latitude, longitude, and optionally address will arrive later
    as a new message from the user — do not guess or invent
    coordinates in the meantime.
    """

    logger.info(
        "GET_LOCATION TOOL CALLED: requesting user location "
        "from frontend"
    )

    return {
        "action": "request_location",
        "message": (
            "Location access has been requested from the user "
            "via the app UI. Do not answer using an assumed, "
            "guessed, or placeholder location. Wait for the user "
            "to provide their actual location in a follow-up "
            "message."
        ),
    }