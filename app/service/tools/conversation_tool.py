from langchain_core.tools import tool

from app.repository.conversation_repo import (
    get_last_10_messages,
)

from app.service.tools.places_tool import (
    search_nearby_places,
)

from app.service.tools.distance_tool import (
    get_distance_bw_2_locations,
    compare_travel_modes,
)

from app.service.tools.search_kb import (
    create_search_knowledge_base_tool,
)

from app.service.tools.geocode_tool import (
    find_location_on_map,
)

from app.service.tools.datetime_tool import (
    get_current_datetime,
)

from app.service.tools.weather_tool import (
    get_weather,
)

from app.service.tools.image_tool import (
    create_image_tool,
    create_document_image_analysis_tool,
)

from app.service.tools.image_generation_tool import (
    generate_image,
)

from app.service.tools.location_tool import (
    get_location,
)

from app.service.tools.tavily_tool import (
    tavily_web_search,
)


# ============================================================
# CREATE CONVERSATION TOOLS
# ============================================================


def create_conversation_tools(
    db,
    user_id: str,
    conversation_id: str | None,
    document_id: str | None = None,
    image_paths: list[str] | None = None,
):
    """
    Create all tools available to the current conversation.

    The application injects:
        - db
        - user_id
        - conversation_id
        - optional document_id
        - optional image_paths

    These values are NOT exposed as arguments to the LLM.

    Available tools:

    1. get_conversation_history
       -> Previous messages from the current conversation.

    2. search_knowledge_base
       -> Uploaded documents / PDFs / images / CSV / Excel /
          DOCX / TXT / Markdown knowledge-base search.

    3. analyze_document_image
       -> Vision analysis of a SPECIFIC image previously extracted
          from an uploaded document/PDF, identified by document_id.
          Always available because it looks up images already stored
          for this user.

    4. get_current_datetime
       -> Current date and time.

    5. get_weather
       -> Current weather information.

    6. get_location
       -> Requests the user's current location by triggering a
          location-selection UI on the frontend.

    7. tavily_web_search
       -> General/current external web search.

    8. get_distance_bw_2_locations
       -> Distance/travel time between two coordinates.

    9. compare_travel_modes
       -> Compare supported travel modes.

    10. search_nearby_places
        -> Search for physical nearby places/POIs.

    11. find_location_on_map
        -> Resolve a specific named place to coordinates.

    12. generate_image
        -> Generate a NEW image from a text prompt.

    13. analyze_image
        -> Analyze image(s) attached directly to the CURRENT
           chat message.
    """

    user_id = str(user_id)

    conversation_id = (
        str(conversation_id)
        if conversation_id is not None
        else None
    )

    if document_id:
        document_id = str(document_id)

    # ========================================================
    # CONVERSATION HISTORY TOOL
    # ========================================================

    @tool
    def get_conversation_history() -> str:
        """
        Fetch recent messages from the current conversation.

        Use this when the current prompt does not contain enough
        context to answer a follow-up question.

        The conversation_id is injected by the application and
        must never be supplied by the LLM.
        """

        if conversation_id is None:
            return (
                "No conversation ID is available, so previous "
                "conversation history cannot be retrieved."
            )

        messages = get_last_10_messages(
            db=db,
            conversation_id=conversation_id,
        )

        if not messages:
            return (
                "No previous conversation history found."
            )

        return "\n".join(
            f"{message.role}: {message.content}"
            for message in messages
        )

    # ========================================================
    # KNOWLEDGE BASE TOOL
    # ========================================================

    search_knowledge_base = (
        create_search_knowledge_base_tool(
            user_id=user_id,
            conversation_id=conversation_id,
            document_id=document_id,
        )
    )

    # ========================================================
    # DOCUMENT IMAGE ANALYSIS TOOL
    #
    # Always available because it looks up previously uploaded
    # document images using the document information.
    # ========================================================

    analyze_document_image = (
        create_document_image_analysis_tool(
            user_id=user_id,
            conversation_id=conversation_id,
        )
    )

    # ========================================================
    # BASE TOOLS
    # ========================================================

    tools = [
        # Conversation / RAG
        get_conversation_history,
        search_knowledge_base,
        analyze_document_image,

        # General utilities
        get_current_datetime,
        get_weather,

        # Location
        get_location,

        # Web
        tavily_web_search,

        # Distance / travel
        get_distance_bw_2_locations,
        compare_travel_modes,

        # Places / maps
        search_nearby_places,
        find_location_on_map,

        # NEW: Image generation
        generate_image,
    ]

    # ========================================================
    # DIRECT IMAGE ANALYSIS TOOL
    #
    # This is ONLY for images attached to the current message.
    # Do not confuse this with generate_image.
    # ========================================================

    valid_image_paths = [
        path
        for path in (image_paths or [])
        if path
    ]

    if valid_image_paths:
        tools.append(
            create_image_tool(
                image_paths=valid_image_paths,
            )
        )

    return tools