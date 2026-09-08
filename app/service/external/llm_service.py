import json
import logging
import os
from typing import Generator, Optional

import requests
from fastapi import HTTPException, status
from langchain_core.prompts import ChatPromptTemplate
from langchain_openrouter import ChatOpenRouter

from app.service.tools.conversation_tool import (
    create_conversation_tools,
)

logger = logging.getLogger(__name__)


# ============================================================
# LLM CONFIGURATION
# ============================================================


LLM_MODEL_NAME = os.getenv(
    "LLM_MODEL",
    "openai/gpt-oss-20b",
).strip()


OPENROUTER_API_KEY = (
    os.getenv("OPENROUTER_API_KEY") or ""
).strip()


REASONING_MODEL_NAME = os.getenv(
    "REASONING_MODEL",
    "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b",
).strip()


CLOUDFLARE_ACCOUNT_ID = (
    os.getenv("CLOUDFLARE_ACCOUNT_ID") or ""
).strip()


CLOUDFLARE_API_TOKEN = (
    os.getenv("CLOUDFLARE_API_TOKEN") or ""
).strip()


CLOUDFLARE_BASE_URL = (
    "https://api.cloudflare.com/client/v4/accounts"
)


llm = ChatOpenRouter(
    model=LLM_MODEL_NAME,
    temperature=0.2,
    api_key=OPENROUTER_API_KEY,
    max_retries=2,
)


# ============================================================
# MULTI-ROUND TOOL CALLING
# ============================================================
#
# Some tasks require a SEQUENCE of tool calls where the second
# tool depends on the result of the first (e.g. resolve a place's
# coordinates with find_location_on_map, THEN call
# get_distance_bw_2_locations with those coordinates). A single
# request/response pass can only execute one round of tool calls,
# so without a loop the model is only ever able to run the FIRST
# step and then has no way to run the second step — it can only
# describe it in prose instead of executing it.
#
# MAX_TOOL_ITERATIONS bounds how many such rounds are allowed per
# user turn, so a misbehaving tool-call loop can never run forever.
# ============================================================

MAX_TOOL_ITERATIONS = int(
    os.getenv(
        "MAX_TOOL_ITERATIONS",
        "4",
    )
)


# ============================================================
# TOOLLESS FAST PATH
# ============================================================
#
# Binding ~11 tool schemas plus the full SYSTEM_PROMPT to every
# single request adds real prompt-processing latency and also
# requires building tool objects (_create_tools) before the LLM
# is even called - overhead a bare greeting or acknowledgement
# ("hi", "ok", "thanks", "bye") never needed in the first place,
# since the model can answer those directly from general
# knowledge with no tool access at all.
#
# Detection is deliberately based on LENGTH, not a keyword list:
# a single-word message with no "?" is treated as too short to be
# a real request. This covers "hi"/"hii"/"heyy"/"thanks"/"ok"/
# "bye"/"gm" etc. regardless of exact wording or language, without
# needing to enumerate them.
#
# TRADE-OFF: a genuine single-word, non-question request (e.g.
# "weather") will also take this fast path and will NOT get tool
# access. This is intentionally conservative (single word only,
# not two words) precisely because skipping tools is riskier than
# skipping the suggestions call. If this trade-off is not
# acceptable, remove or tighten this fast path rather than trying
# to special-case individual words.
# ============================================================


def _is_toolless_fast_path(
    question: str,
) -> bool:
    """Return True only for clearly conversational one-word messages.

    Do not use a generic one-word heuristic here because legitimate
    tool queries such as ``weather`` or ``restaurants`` must still
    receive tool access.
    """
    if not question:
        return False

    normalized = question.strip().lower().strip(".!?,")

    conversational_messages = {
        "hi",
        "hii",
        "hello",
        "hey",
        "heyy",
        "thanks",
        "thankyou",
        "ok",
        "okay",
        "bye",
        "goodbye",
        "gm",
        "goodmorning",
        "goodnight",
    }

    return normalized in conversational_messages


# ============================================================
# CLOUDFLARE REASONING ADAPTER
# ============================================================


_ROLE_MAP = {
    "system": "system",
    "human": "user",
    "user": "user",
    "ai": "assistant",
    "assistant": "assistant",
    "tool": "user",
}


def _reasoning_messages_to_cf_messages(messages):

    cf_messages = []

    for message in messages:

        if isinstance(message, tuple):

            role, content = message

        else:

            role = getattr(
                message,
                "type",
                "human",
            )

            content = getattr(
                message,
                "content",
                str(message),
            )

        # --------------------------------------------------------
        # LangChain content can occasionally be structured.
        # Cloudflare expects text here.
        # --------------------------------------------------------

        if not isinstance(content, str):

            try:

                content = json.dumps(
                    content,
                    ensure_ascii=False,
                )

            except Exception:

                content = str(content)

        cf_messages.append(
            {
                "role": _ROLE_MAP.get(
                    role,
                    "user",
                ),
                "content": content,
            }
        )

    return cf_messages


class _ReasoningChunk:

    __slots__ = (
        "content",
        "additional_kwargs",
    )

    def __init__(
        self,
        content: str,
    ):

        self.content = content
        self.additional_kwargs = {}


class _ReasoningResponse:

    def __init__(
        self,
        content: str,
    ):

        self.content = content


REASONING_MAX_TOKENS = int(
    os.getenv(
        "REASONING_MAX_TOKENS",
        "4096",
    )
)


class _CloudflareReasoningLLM:

    def __init__(
        self,
        account_id: str,
        api_token: str,
        model: str,
    ):

        self._account_id = account_id
        self._api_token = api_token
        self._model = model

        self._endpoint = (
            f"{CLOUDFLARE_BASE_URL}/"
            f"{account_id}/ai/run/{model}"
        )

    def _request(
        self,
        messages,
        stream: bool,
    ):

        if not self._account_id:

            raise RuntimeError(
                "CLOUDFLARE_ACCOUNT_ID is not configured"
            )

        if not self._api_token:

            raise RuntimeError(
                "CLOUDFLARE_API_TOKEN is not configured"
            )

        headers = {
            "Authorization": (
                f"Bearer {self._api_token}"
            ),
            "Content-Type": "application/json",
        }

        payload = {
            "messages": (
                _reasoning_messages_to_cf_messages(
                    messages
                )
            ),
            "stream": stream,
            "max_tokens": REASONING_MAX_TOKENS,
        }

        response = requests.post(
            self._endpoint,
            headers=headers,
            json=payload,
            stream=stream,
            timeout=120,
        )

        if not response.ok:

            raise RuntimeError(
                "Cloudflare Workers AI request failed "
                f"(status={response.status_code}): "
                f"{response.text}"
            )

        return response

    def invoke(
        self,
        messages,
    ):

        response = self._request(
            messages,
            stream=False,
        )

        data = response.json()

        result = data.get(
            "result",
            {},
        )

        text = result.get(
            "response",
            "",
        )

        return _ReasoningResponse(
            text or "",
        )

    def stream(
        self,
        messages,
    ):

        response = self._request(
            messages,
            stream=True,
        )

        for raw_line in response.iter_lines():

            if not raw_line:
                continue

            # ----------------------------------------------------
            # Handle normal SSE data lines.
            # ----------------------------------------------------

            if not raw_line.startswith(
                b"data:"
            ):

                continue

            payload_bytes = raw_line[
                len(b"data:"):
            ].strip()

            if not payload_bytes:
                continue

            if payload_bytes == b"[DONE]":
                break

            try:

                data = json.loads(
                    payload_bytes
                )

            except json.JSONDecodeError:

                logger.warning(
                    "REASONING STREAM: "
                    "skipping unparseable line"
                )

                continue

            # ----------------------------------------------------
            # FIX: Cloudflare Workers AI's STREAMING (SSE) payload
            # puts the generated token at the TOP LEVEL, e.g.
            #     {"response": "some token"}
            # The nested {"result": {"response": ...}} shape only
            # applies to the NON-STREAMING /ai/run JSON response
            # (see invoke() above). Previously this method only
            # ever checked data["result"]["response"], which for
            # streamed responses is always {} -> None -> every
            # chunk was silently dropped -> the whole stream
            # yielded nothing, surfacing as "REASONING EMPTY
            # RESULT" in the logs even though Cloudflare was
            # responding successfully.
            #
            # Both shapes are now checked, nested first (kept for
            # forward/backward compatibility in case a future
            # Cloudflare model nests it), falling back to the
            # top-level key that streaming actually uses.
            # ----------------------------------------------------

            result = data.get(
                "result",
                {},
            )

            token = result.get(
                "response"
            )

            if token is None:

                token = data.get(
                    "response"
                )

            if token:

                yield _ReasoningChunk(
                    token
                )


reasoning_llm = _CloudflareReasoningLLM(
    account_id=CLOUDFLARE_ACCOUNT_ID,
    api_token=CLOUDFLARE_API_TOKEN,
    model=REASONING_MODEL_NAME,
)


# ============================================================
# IMAGE HELPERS
# ============================================================


def _is_image_content_type(
    content_type,
) -> bool:

    if not content_type:
        return False

    content_type = str(
        content_type
    )

    return (
        content_type == "image"
        or content_type.startswith(
            "image/"
        )
    )


# ============================================================
# SYSTEM PROMPT
# ============================================================


SYSTEM_PROMPT = """
You are a helpful AI assistant.

You have access to:

1. Conversation history
2. Conversation history tool
3. Uploaded-document knowledge-base search tool
4. Document image analysis tool
5. Current date and time tool
6. Weather tool
7. Get user location tool
8. Search nearby places tool
9. Direct image analysis tool
10. Web search tool
11. Find location on map tool
12. Distance-between-two-locations tool
13. Compare-travel-modes tool
14. Image generation tool

============================================================
TOOL ROUTING — IMPORTANT
============================================================

Before calling any tool, determine the user's PRIMARY INTENT.
Do not select a tool merely because one word in the query matches
a tool description. Understand what the user is actually asking for.

The phrase "near me" describes a LOCATION CONSTRAINT. It does NOT
by itself determine which tool must be used. The type of information
the user wants determines the tool.

Use the MINIMUM number of tools necessary. Do not call unrelated
tools. Do not call the same tool repeatedly unless the next call is
actually needed.

If the query can be answered reliably without a tool, answer directly.

============================================================
NEARBY PLACES VS WEB SEARCH
============================================================

1. SEARCH_NEARBY_PLACES

Use search_nearby_places when the user wants to DISCOVER ACTUAL
PHYSICAL PLACES, BUSINESSES, or SERVICES around a location.

Examples:
- "restaurants near me"
- "cafes near me"
- "hotels near me"
- "hospitals near me"
- "pharmacies near me"
- "banks near me"
- "ATMs near me"
- "parks near me"
- "malls near me"
- "temples near me"
- "petrol pumps near me"
- "gas stations near me"
- "train stations near me"
- "find Starbucks near me"

These queries are asking for nearby physical places.

2. TAVILY_WEB_SEARCH

Use tavily_web_search when the user wants CURRENT, EXTERNAL,
SEARCHABLE INFORMATION or LISTINGS rather than a nearby-place
category supported by search_nearby_places.

IMPORTANT: Use web search for property/real-estate queries even
when the user says "near me".

Examples:
- "properties near me" -> tavily_web_search
- "property for sale near me" -> tavily_web_search
- "houses for sale near me" -> tavily_web_search
- "apartments for rent near me" -> tavily_web_search
- "flats for sale near me" -> tavily_web_search
- "land for sale near me" -> tavily_web_search
- "plots near me" -> tavily_web_search
- "commercial property near me" -> tavily_web_search
- "real estate near me" -> tavily_web_search
- "jobs near me" -> tavily_web_search
- "cars for sale near me" -> tavily_web_search
- "used bikes near me" -> tavily_web_search

Do NOT force these queries into search_nearby_places simply because
they contain "near me".

If a query asks for a category that search_nearby_places does not
support, use tavily_web_search when current/external information
is needed.

3. SPECIFIC PLACE + WEB INFORMATION

If the user asks for current information about a specific place,
business, property, listing, event, price, availability, review,
or other changing information, use tavily_web_search when the
information is external/current.

4. MAP

Use find_location_on_map when the user wants ONE SPECIFIC PLACE
located or shown on a map.

If a web search finds a property/business/listing and the user then
asks to show that result on a map, use find_location_on_map for the
specific result if its location can be resolved.

============================================================
LOCATION ROUTING
============================================================

Always inspect the User Location section before location-dependent
tool calls.

If the user's location is already known, DO NOT call get_location.
Use the supplied exact coordinates for location-aware tools.

If the query requires the user's current location and it is NOT
known, call get_location first.

This applies to BOTH nearby-place searches AND web searches where
"near me" changes the search meaning, such as:
- properties near me
- houses for sale near me
- jobs near me

After location is supplied, continue the original request. Do not
replace the user's original intent with a generic nearby-place search.

Never guess the user's location.

Location persists for this conversation when the User Location section
says it is already known. Do not ask for it again unless the user
requests a different location or the section says it is not known.

============================================================
KNOWLEDGE BASE / RAG
============================================================

Use search_knowledge_base when the answer may be present in the
user's uploaded documents or knowledge base.

If an uploaded document is available for the current conversation,
search the knowledge base FIRST for factual questions that could
reasonably be answered from that document.

The user does not need to explicitly mention the document.

When relevant knowledge-base content is returned, treat it as the
source of truth. Do not invent document facts.

If the knowledge base has no relevant information, clearly say that
the information was not found in the uploaded knowledge base rather
than fabricating an answer.

If the user explicitly asks for external/current information, web
search may be used even when documents exist.

============================================================
WEB SEARCH
============================================================

Use tavily_web_search for:
- current or recent information
- news
- live/public web facts
- external information
- properties and real-estate listings
- jobs and opportunities
- products or listings
- current prices or availability
- information not answerable from the conversation or knowledge base

Do NOT use tavily_web_search as a replacement for dedicated tools
when the user is clearly asking for weather, current date/time,
exact route distance, travel-mode comparison, or supported nearby
physical places.

However, when the user is asking for an external searchable domain
that is NOT supported by search_nearby_places (for example property
listings), use tavily_web_search even if the query contains "near me".

============================================================
CURRENT DATE / TIME
============================================================

Use get_current_datetime when the user asks for the current date,
current time, today's date/time, or similar. Never guess.

============================================================
WEATHER
============================================================

Use get_weather for current weather, temperature, forecast, rain,
conditions, or similar weather questions. Never invent current
weather data.

============================================================
PLACES
============================================================

Use search_nearby_places ONLY for nearby physical-place discovery.

Allowed canonical categories:
attraction, cafe, restaurant, food, pharmacy, hospital, hotel, bank,
atm, park, mall, temple, fuel, station.

Map natural language to the closest supported category:
- coffee shop -> cafe
- restaurant to eat -> food or restaurant
- medicine shop -> pharmacy
- petrol pump / gas station -> fuel
- train station -> station

Do not invent unsupported categories.

For unsupported nearby domains such as property listings, jobs, cars
for sale, land, apartments, or other web-searchable listings, use
tavily_web_search instead.

If the user asks "what is around me?" without a clear category, use
search_nearby_places with no forced category rather than inventing one.

For follow-ups such as "show me another one", "which is closest?",
or "something else nearby", use conversation history to preserve the
previous nearby-place intent and category when appropriate.

Never fabricate business names, addresses, coordinates, distances,
ratings, prices, or opening hours. Only report information returned
by the relevant tool.

============================================================
MAP
============================================================

Use find_location_on_map ONLY to geocode/show ONE specific named
place on a map.

Do not use it as a substitute for nearby-place discovery.
Do not use it as a substitute for distance calculation.
Do not invent map URLs or coordinates.

============================================================
DISTANCE / TRAVEL
============================================================

Use get_distance_bw_2_locations when the user asks how far two
locations are apart or asks for travel time using ONE specific mode.

Use compare_travel_modes when the user asks to compare TWO OR MORE
travel modes.

Both require real coordinates. Never guess coordinates.

If the destination is a named place whose coordinates are unknown,
first resolve it with find_location_on_map (or another relevant tool
when appropriate), then perform the actual distance/travel calculation.

Supported travel modes are driving, walking, and cycling. Do not
invent results for unsupported modes.

============================================================
IMAGES
============================================================

If an image was attached directly to the current message and the
user asks about it, use analyze_image.

For an image inside an uploaded document, use search_knowledge_base
and analyze_document_image as appropriate.

Never invent image URLs or document IDs.

============================================================
IMAGE GENERATION
============================================================

Use generate_image ONLY when the user explicitly asks to create,
generate, draw, make, or produce a NEW image.

Examples:
- "Generate an image of a futuristic city"
- "Create a picture of a mountain"
- "Draw a cartoon robot"

Do NOT use generate_image to analyze an existing image or find an
existing image on the web.

When generation succeeds, use the exact URL returned by the tool.
Never invent or modify the generated image URL.

============================================================
MULTI-TOOL EXECUTION
============================================================

You may call multiple tools across multiple rounds when the task
requires them. Execute the required sequence instead of describing
a plan to the user.

Examples:

"Find restaurants near me and show the best one on a map"
-> get_location (only if needed)
-> search_nearby_places
-> find_location_on_map for the selected specific result

"Find properties near me"
-> get_location (only if needed)
-> tavily_web_search

"How far is the airport from me?"
-> get_location (only if needed)
-> find_location_on_map for the airport if coordinates are unknown
-> get_distance_bw_2_locations

"Compare driving and walking to the airport"
-> get_location (only if needed)
-> resolve destination if needed
-> compare_travel_modes

Do not call a tool just because it is available.
Do not call nearby_places for every query containing "near me".
Do not call web search for queries that have a dedicated reliable
tool unless the user explicitly asks for external/current web data.

============================================================
FINAL ANSWER
============================================================

Use the actual tool results as the source of truth.
Never fabricate missing data.
If a tool fails, explain the limitation briefly and use another
available method only when appropriate.
Do not mention internal tool names, routing rules, hidden reasoning,
or chain-of-thought to the user.
Keep the final answer clear, useful, and concise.
"""


# ============================================================
# LOCATION CONTEXT
# ============================================================


def _build_location_context(
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    address: Optional[str] = None,
) -> str:

    if (
        latitude is not None
        and longitude is not None
    ):

        location_line = (
            "The user's current location is ALREADY KNOWN: "
            f"latitude={latitude}, "
            f"longitude={longitude}."
        )

        if address:

            location_line += (
                f" Address='{address}'."
            )

        location_line += (
            " Do NOT call get_location. "
            "Use these exact coordinates directly when "
            "calling nearby-place or distance/travel tools."
        )

        return location_line

    return (
        "The user's current location is NOT known. "
        "If the question requires the user's current "
        "location, call get_location first."
    )


# ============================================================
# BUILD MESSAGES
# ============================================================


def _build_messages(
    question: str,
    chat_history: Optional[list[dict]] = None,
    document_available: bool = False,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    address: Optional[str] = None,
):

    chat_history = (
        chat_history or []
    )

    recent_history = chat_history[-10:]

    history_text = "\n".join(
        f"{message.get('role', '')}: "
        f"{message.get('content', '')}"
        for message in recent_history
    )

    if not history_text:

        history_text = (
            "No previous conversation history."
        )

    if document_available:

        document_context = (
            "YES. This turn is scoped to one specific "
            "uploaded document. If the current question "
            "could be answered from it, use "
            "search_knowledge_base first."
        )

    else:

        document_context = (
            "Documents MAY be available in the knowledge "
            "base for this user. This includes global "
            "documents and documents belonging to this "
            "conversation. No single document_id is "
            "pre-selected. If the current question could "
            "reasonably be answered from an uploaded "
            "document, use search_knowledge_base first."
        )

    location_context = _build_location_context(
        latitude=latitude,
        longitude=longitude,
        address=address,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SYSTEM_PROMPT,
            ),
            (
                "human",
                """
Conversation History:

{history}

Uploaded Document Available:

{document_context}

User Location:

{location_context}

Current User Question:

{question}
""",
            ),
        ]
    )

    return prompt.format_messages(
        history=history_text,
        document_context=document_context,
        location_context=location_context,
        question=question,
    )


# ============================================================
# CREATE TOOLS
# ============================================================


def _create_tools(
    db=None,
    user_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
    document_id: Optional[str] = None,
    image_paths: Optional[list[str]] = None,
):

    if (
        db is None
        or user_id is None
        or conversation_id is None
    ):

        return []

    return create_conversation_tools(
        db=db,
        user_id=str(user_id),
        conversation_id=str(conversation_id),
        document_id=(
            str(document_id)
            if document_id
            else None
        ),
        image_paths=image_paths,
    )


# ============================================================
# TOOL BINDING
# ============================================================


def _bind_tools(
    tools: list,
):

    if not tools:
        return llm

    return llm.bind_tools(
        tools
    )


def _get_tool(
    tools: list,
    tool_name: str,
):

    return next(
        (
            tool
            for tool in tools
            if tool.name == tool_name
        ),
        None,
    )


# ============================================================
# EXECUTE TOOLS
# ============================================================


def _execute_tool_calls(
    tools: list,
    tool_calls: list,
    conversation_id: Optional[str],
    log_prefix: str = "",
    known_latitude: Optional[float] = None,
    known_longitude: Optional[float] = None,
):

    tool_messages = []

    collected_images = []

    location_request = None

    map_location = None

    for tool_call in tool_calls:

        tool_name = tool_call["name"]

        tool_args = tool_call.get(
            "args",
            {},
        )

        # ====================================================
        # LOCATION SAFETY NET
        # ====================================================

        if (
            tool_name == "get_location"
            and known_latitude is not None
            and known_longitude is not None
        ):

            logger.info(
                "%sLOCATION ALREADY KNOWN: "
                "conversation_id=%s",
                log_prefix,
                conversation_id,
            )

            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": (
                        "The user's location is already known. "
                        f"latitude={known_latitude}, "
                        f"longitude={known_longitude}. "
                        "Use these coordinates directly."
                    ),
                }
            )

            continue

        selected_tool = _get_tool(
            tools=tools,
            tool_name=tool_name,
        )

        if selected_tool is None:

            raise RuntimeError(
                f"Requested tool not found: {tool_name}"
            )

        logger.info(
            "%sTOOL EXECUTING: tool=%s args=%s "
            "conversation_id=%s",
            log_prefix,
            tool_name,
            tool_args,
            conversation_id,
        )

        try:

            tool_result = selected_tool.invoke(
                tool_args
            )

        except Exception:

            logger.exception(
                "%sTOOL FAILED: tool=%s "
                "conversation_id=%s",
                log_prefix,
                tool_name,
                conversation_id,
            )

            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": (
                        f"The '{tool_name}' tool failed. "
                        "Use any other available information "
                        "to answer the user."
                    ),
                }
            )

            continue

        logger.info(
            "%sTOOL RESULT: tool=%s "
            "conversation_id=%s",
            log_prefix,
            tool_name,
            conversation_id,
        )

        # ====================================================
        # IMAGE RESULTS
        # ====================================================

        if (
            tool_name == "search_knowledge_base"
            and isinstance(tool_result, list)
        ):

            for item in tool_result:

                if not isinstance(
                    item,
                    dict,
                ):
                    continue

                content_type = item.get(
                    "content_type"
                )

                if (
                    _is_image_content_type(
                        content_type
                    )
                    and item.get("document_id")
                ):

                    image_document_id = str(
                        item.get(
                            "document_id"
                        )
                    )

                    collected_images.append(
                        {
                            "document_id": item.get(
                                "document_id"
                            ),
                            "parent_document_id": (
                                item.get(
                                    "parent_document_id"
                                )
                                or item.get(
                                    "document_id"
                                )
                            ),
                            "filename": item.get(
                                "filename"
                            ),
                            "url": (
                                f"/documents/"
                                f"{image_document_id}/file"
                            ),
                        }
                    )

        elif (
            tool_name == "analyze_document_image"
            and isinstance(tool_result, dict)
            and tool_result.get("success")
            and tool_result.get("document_id")
        ):

            image_document_id = str(
                tool_result.get(
                    "document_id"
                )
            )

            collected_images.append(
                {
                    "document_id": tool_result.get(
                        "document_id"
                    ),
                    "parent_document_id": (
                        tool_result.get(
                            "parent_document_id"
                        )
                        or tool_result.get(
                            "document_id"
                        )
                    ),
                    "filename": tool_result.get(
                        "filename"
                    ),
                    "url": (
                        f"/documents/"
                        f"{image_document_id}/file"
                    ),
                }
            )

        # ====================================================
        # GENERATED IMAGE
        # ====================================================

        elif (
            tool_name == "generate_image"
            and isinstance(tool_result, dict)
            and tool_result.get("success")
            and tool_result.get("url")
        ):
            collected_images.append(
                {
                    "type": "generated",
                    "filename": tool_result.get(
                        "filename"
                    ),
                    "url": tool_result.get(
                        "url"
                    ),
                    "prompt": tool_result.get(
                        "prompt"
                    ),
                    "model": tool_result.get(
                        "model"
                    ),
                }
            )

        # ====================================================
        # LOCATION REQUEST
        # ====================================================

        elif (
            tool_name == "get_location"
            and isinstance(tool_result, dict)
            and tool_result.get("action")
            == "request_location"
        ):

            location_request = tool_result

        # ====================================================
        # MAP LOCATION
        # ====================================================

        elif (
            tool_name == "find_location_on_map"
            and isinstance(tool_result, dict)
            and tool_result.get("success")
            and tool_result.get("action")
            == "show_map"
        ):

            map_location = tool_result

        # ====================================================
        # TOOL MESSAGE
        # ====================================================

        tool_messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": str(tool_result),
            }
        )

    return (
        tool_messages,
        collected_images,
        location_request,
        map_location,
    )


# ============================================================
# REASONING DECISION
# ============================================================
#
# Reasoning is normally triggered whenever any tool was called,
# since tool output usually needs to be synthesized into a
# clean final answer.
#
# EXCEPTION: if the ONLY tool call across all rounds this turn is
# "get_location", there is no real data to reason over yet — the
# model is just asking the frontend for the user's coordinates.
# Sending that to the reasoning model wastes a call and is a
# likely source of "REASONING EMPTY RESULT" (nothing meaningful
# to summarize). In that case we skip reasoning entirely.
#
# Every OTHER location-related tool (search_nearby_places,
# get_distance_bw_2_locations, compare_travel_modes,
# find_location_on_map) still goes through reasoning as before,
# since those calls return real data that benefits from being
# synthesized into a clean answer.
# ============================================================


LOCATION_TOOL_NAMES = {
    "search_nearby_places",
    "get_distance_bw_2_locations",
    "compare_travel_modes",
    "find_location_on_map",
}


# ============================================================
# TOOL -> USER-FACING STAGE DESCRIPTION
#
# Used only to build a friendlier, multi-step "thinking" status
# trail in the UI (see _build_stage_message below). This is NOT
# raw chain-of-thought — just a human-readable label for which
# kind of tool result the reasoning step is working with, so the
# UI can show a believable multi-stage process (Understanding ->
# Reviewing <this> -> Analyzing -> Preparing answer) instead of a
# single static line repeated forever.
# ============================================================

TOOL_STAGE_DESCRIPTIONS = {
    "get_conversation_history": "Reviewing the conversation so far",
    "search_knowledge_base": "Searching the uploaded documents",
    "analyze_document_image": "Looking closely at the document image",
    "analyze_image": "Looking closely at the attached image",
    "generate_image": "Creating the image",
    "get_current_datetime": "Checking the current date and time",
    "get_weather": "Checking the weather",
    "get_location": "Getting your location",
    "tavily_web_search": "Searching the web",
    "get_distance_bw_2_locations": "Calculating the distance",
    "compare_travel_modes": "Comparing travel options",
    "search_nearby_places": "Looking for nearby places",
    "find_location_on_map": "Locating that place on the map",
}


def _build_stage_message(
    tool_calls: list,
) -> Optional[str]:
    """
    Build a short, human-readable "what am I working with right
    now" status line from the tools that were actually called this
    turn, e.g. "Reviewing nearby places and distance results...".
    Falls back to a generic line if none of the tool names are
    recognised. Never reveals raw tool arguments/results or model
    chain-of-thought — just the category of work in progress.
    """

    if not tool_calls:
        return None

    labels = []

    for tool_call in tool_calls:

        tool_name = tool_call.get("name")

        label = TOOL_STAGE_DESCRIPTIONS.get(
            tool_name
        )

        if label and label not in labels:

            labels.append(label)

    if not labels:
        return "Reviewing the retrieved information"

    if len(labels) == 1:
        return f"{labels[0]}..."

    return (
        ", ".join(labels[:-1])
        + f" and {labels[-1]}..."
    )


def _should_use_reasoning(
    tool_calls: list,
    collected_images: list,
    extra_messages: list,
    conversation_id: Optional[str] = None,
) -> bool:

    if not tool_calls:
        return False

    tool_names = {
        tool_call.get("name")
        for tool_call in tool_calls
    }

    # ------------------------------------------------------------
    # Sirf get_location call hua ho (koi aur tool nahi) — iska
    # matlab abhi LLM sirf user ka location maang raha hai, koi
    # actual data reason karne ke liye maujood nahi hai.
    # Is case mein reasoning skip kar do.
    # ------------------------------------------------------------

    if tool_names == {"get_location"}:

        logger.info(
            "REASONING SKIPPED (get_location only): "
            "conversation_id=%s",
            conversation_id,
        )

        return False

    return True


# ============================================================
# SAFE THINKING STREAM SPLITTER
#
# We intentionally DO NOT expose raw chain-of-thought.
#
# Instead, reasoning output is consumed internally and the
# frontend receives a short status message.
# ============================================================


def _stream_with_thinking_split(
    model_stream,
):

    state = "answer"

    pending = ""

    for chunk in model_stream:

        content = getattr(
            chunk,
            "content",
            "",
        ) or ""

        if not content:
            continue

        pending += content

        while pending:

            if state == "answer":

                start_index = pending.find(
                    "<think>"
                )

                if start_index == -1:

                    safe_length = max(
                        0,
                        len(pending) - 7,
                    )

                    if safe_length:

                        yield {
                            "type": "answer",
                            "content": pending[
                                :safe_length
                            ],
                        }

                        pending = pending[
                            safe_length:
                        ]

                    break

                if start_index:

                    yield {
                        "type": "answer",
                        "content": pending[
                            :start_index
                        ],
                    }

                pending = pending[
                    start_index + 7:
                ]

                state = "thinking"

            else:

                end_index = pending.find(
                    "</think>"
                )

                if end_index == -1:

                    safe_length = max(
                        0,
                        len(pending) - 8,
                    )

                    if safe_length:

                        # Do not expose reasoning text.
                        pending = pending[
                            safe_length:
                        ]

                    break

                pending = pending[
                    end_index + 8:
                ]

                state = "answer"

    if pending and state == "answer":

        yield {
            "type": "answer",
            "content": pending,
        }


# ============================================================
# STRIP THINKING
# ============================================================


def _strip_thinking(
    text: str,
) -> str:

    if not text:
        return text

    result = []

    remaining = text

    while True:

        start_index = remaining.find(
            "<think>"
        )

        if start_index == -1:

            result.append(
                remaining
            )

            break

        result.append(
            remaining[:start_index]
        )

        end_index = remaining.find(
            "</think>",
            start_index + 7,
        )

        if end_index == -1:
            break

        remaining = remaining[
            end_index + 8:
        ]

    return "".join(
        result
    ).strip()


# ============================================================
# REASONING MESSAGE BUILDER
# ============================================================


def _build_reasoning_messages(
    base_messages: list,
    tool_messages: list,
):

    reasoning_instruction = (
        "Using the question, conversation context, and "
        "tool results above, produce one clear final answer.\n\n"
        "Rules:\n"
        "- Use only relevant tool information.\n"
        "- Ignore irrelevant retrieval results.\n"
        "- Do not invent facts.\n"
        "- Do not mention internal tools or reasoning steps.\n"
        "- Do not reveal hidden chain-of-thought.\n"
        "- Keep the answer natural and concise.\n"
        "- If the tool results contain a relevant image URL, "
        "use that URL exactly as returned by the tool.\n"
        "- If an image was generated successfully, refer to the "
        "generated image naturally and never invent or modify "
        "its URL.\n"
    )

    return (
        base_messages
        + tool_messages
        + [
            (
                "human",
                reasoning_instruction,
            )
        ]
    )


# ============================================================
# GENERATE ANSWER
# ============================================================


def generate_answer(
    question: str,
    chat_history: Optional[list[dict]] = None,
    db=None,
    user_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
    images_output: Optional[list] = None,
    document_id: Optional[str] = None,
    image_paths: Optional[list[str]] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    address: Optional[str] = None,
):

    try:

        messages = _build_messages(
            question=question,
            chat_history=chat_history,
            document_available=bool(
                document_id
            ),
            latitude=latitude,
            longitude=longitude,
            address=address,
        )

        # ========================================================
        # TOOLLESS FAST PATH
        #
        # Skip tool creation/binding entirely for bare greetings
        # and acknowledgements - see _is_toolless_fast_path() above
        # for the exact rule and its trade-off.
        # ========================================================

        if _is_toolless_fast_path(
            question
        ):

            logger.info(
                "TOOLLESS FAST PATH: "
                "conversation_id=%s question=%s",
                conversation_id,
                question,
            )

            fast_response = llm.invoke(
                messages
            )

            return _strip_thinking(
                fast_response.content
            )

        tools = _create_tools(
            db=db,
            user_id=user_id,
            conversation_id=conversation_id,
            document_id=document_id,
            image_paths=image_paths,
        )

        llm_with_tools = _bind_tools(
            tools
        )

        # ========================================================
        # MULTI-ROUND TOOL CALLING LOOP
        #
        # Runs up to MAX_TOOL_ITERATIONS rounds so that a task
        # requiring a chain of tool calls (e.g. find_location_on_map
        # -> get_distance_bw_2_locations) can actually complete,
        # instead of the model only ever getting to run the first
        # tool and then being forced to describe the remaining
        # steps in prose. If the model responds with no further
        # tool_calls in any round, that round's response is treated
        # as final (same behavior as before for the single-round
        # case).
        # ========================================================

        current_messages = list(messages)

        all_tool_messages = []

        all_tool_calls = []

        response = None

        for _iteration in range(MAX_TOOL_ITERATIONS):

            try:

                response = llm_with_tools.invoke(
                    current_messages
                )

            except Exception:

                # ----------------------------------------------------
                # A later round (after we already have at least one
                # round of real tool results, e.g. a web search) can
                # fail/timeout because the tool result made the
                # request too large/slow for the provider. Rather
                # than losing the tool data we already fetched, fall
                # back to answering with what we have. If this is
                # the VERY FIRST round (no tool results yet at all),
                # there is nothing useful to fall back to, so the
                # original behavior (raise, handled by the outer
                # except block below) is preserved.
                # ----------------------------------------------------

                if all_tool_calls:

                    logger.exception(
                        "TOOL ROUND FAILED (iteration=%s), "
                        "falling back to results gathered so far: "
                        "conversation_id=%s",
                        _iteration,
                        conversation_id,
                    )

                    break

                raise

            if not response.tool_calls:
                break

            logger.info(
                "LLM TOOL CALLS: tools=%s "
                "conversation_id=%s",
                [
                    tool_call["name"]
                    for tool_call in response.tool_calls
                ],
                conversation_id,
            )

            all_tool_calls.extend(
                response.tool_calls
            )

            (
                extra_messages,
                collected_images,
                _location_request,
                _map_location,
            ) = _execute_tool_calls(
                tools=tools,
                tool_calls=response.tool_calls,
                conversation_id=conversation_id,
                known_latitude=latitude,
                known_longitude=longitude,
            )

            if images_output is not None:

                images_output.extend(
                    collected_images
                )

            round_messages = (
                [response] + extra_messages
            )

            all_tool_messages.extend(
                round_messages
            )

            current_messages = (
                current_messages + round_messages
            )

        if response is None:

            raise RuntimeError(
                "Unable to obtain a response from the LLM"
            )

        if not all_tool_calls:

            return _strip_thinking(
                response.content
            )

        # ========================================================
        # REASONING
        # ========================================================

        if _should_use_reasoning(
            tool_calls=all_tool_calls,
            collected_images=[],
            extra_messages=all_tool_messages,
            conversation_id=conversation_id,
        ):

            try:

                logger.info(
                    "REASONING START: model=%s "
                    "conversation_id=%s",
                    REASONING_MODEL_NAME,
                    conversation_id,
                )

                reasoning_messages = (
                    _build_reasoning_messages(
                        base_messages=messages,
                        tool_messages=all_tool_messages,
                    )
                )

                reasoning_response = (
                    reasoning_llm.invoke(
                        reasoning_messages
                    )
                )

                final_answer = _strip_thinking(
                    reasoning_response.content
                )

                if final_answer:

                    return final_answer

            except Exception:

                logger.exception(
                    "REASONING FAILED, "
                    "falling back to main LLM: "
                    "conversation_id=%s",
                    conversation_id,
                )

        # ========================================================
        # MAIN LLM FINAL ANSWER
        #
        # IMPORTANT:
        # Tools are intentionally NOT bound here.
        # ========================================================

        final_response = llm.invoke(
            messages + all_tool_messages
        )

        return _strip_thinking(
            final_response.content
        )

    except HTTPException:

        raise

    except Exception as exc:

        logger.exception(
            "LLM RESPONSE ERROR: "
            "conversation_id=%s error=%s",
            conversation_id,
            str(exc),
        )

        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Unable to generate response from LLM",
        ) from exc


# ============================================================
# PARSE STREAMING TOOL CALLS
# ============================================================


def _parse_tool_calls(
    tool_call_chunks: list,
):

    if not tool_call_chunks:

        return []

    tool_calls = {}

    for chunk in tool_call_chunks:

        index = chunk.get(
            "index",
            0,
        )

        if index not in tool_calls:

            tool_calls[index] = {
                "id": "",
                "name": "",
                "args": "",
            }

        tool_calls[index]["id"] += (
            chunk.get("id") or ""
        )

        tool_calls[index]["name"] += (
            chunk.get("name") or ""
        )

        tool_calls[index]["args"] += (
            chunk.get("args") or ""
        )

    parsed_calls = []

    for tool_call in tool_calls.values():

        args_text = (
            tool_call["args"].strip()
        )

        if args_text:

            try:

                args = json.loads(
                    args_text
                )

            except json.JSONDecodeError as exc:

                raise RuntimeError(
                    "Unable to parse tool arguments"
                ) from exc

        else:

            args = {}

        parsed_calls.append(
            {
                "id": tool_call["id"],
                "name": tool_call["name"],
                "args": args,
            }
        )

    return parsed_calls


# ============================================================
# GENERATE STREAMING ANSWER
# ============================================================


def generate_answer_stream(
    question: str,
    chat_history: Optional[list[dict]] = None,
    db=None,
    user_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
    images_output: Optional[list] = None,
    document_id: Optional[str] = None,
    image_paths: Optional[list[str]] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    address: Optional[str] = None,
) -> Generator[dict, None, None]:

    try:

        messages = _build_messages(
            question=question,
            chat_history=chat_history,
            document_available=bool(
                document_id
            ),
            latitude=latitude,
            longitude=longitude,
            address=address,
        )

        # ========================================================
        # TOOLLESS FAST PATH
        #
        # Skip tool creation/binding entirely for bare greetings
        # and acknowledgements - see _is_toolless_fast_path() above
        # for the exact rule and its trade-off.
        # ========================================================

        if _is_toolless_fast_path(
            question
        ):

            logger.info(
                "TOOLLESS FAST PATH: "
                "conversation_id=%s question=%s",
                conversation_id,
                question,
            )

            for chunk in llm.stream(
                messages
            ):

                content = getattr(
                    chunk,
                    "content",
                    None,
                )

                if content:

                    yield {
                        "type": "answer",
                        "content": content,
                    }

            return

        tools = _create_tools(
            db=db,
            user_id=user_id,
            conversation_id=conversation_id,
            document_id=document_id,
            image_paths=image_paths,
        )

        llm_with_tools = _bind_tools(
            tools
        )

        logger.info(
            "LLM STREAM START: "
            "conversation_id=%s tools=%s",
            conversation_id,
            [
                tool.name
                for tool in tools
            ],
        )

        # ========================================================
        # MULTI-ROUND TOOL CALLING LOOP (STREAMING)
        #
        # Same reasoning as generate_answer() above: without this
        # loop, a chained task like "find place -> then calculate
        # distance to it" could only ever execute the FIRST tool
        # call and then had no way to run the second one, so the
        # model was forced to explain the remaining steps in prose
        # instead of actually doing them.
        #
        # Each round streams the model's response, parses any tool
        # calls, executes them, and (if there are more tool calls)
        # feeds the results back in for another round. If a round
        # produces no tool calls, that round's streamed content is
        # the final answer and is streamed straight to the caller
        # exactly like the original NO TOOLS case.
        # ========================================================

        current_messages = list(messages)

        all_tool_messages = []

        all_tool_calls = []

        for _iteration in range(MAX_TOOL_ITERATIONS):

            streamed_chunks = []

            tool_call_chunks = []

            round_failed = False

            try:

                for chunk in llm_with_tools.stream(
                    current_messages
                ):

                    streamed_chunks.append(
                        chunk
                    )

                    current_tool_chunks = getattr(
                        chunk,
                        "tool_call_chunks",
                        None,
                    )

                    if current_tool_chunks:

                        tool_call_chunks.extend(
                            current_tool_chunks
                        )

            except Exception:

                # ----------------------------------------------------
                # Same reasoning as the non-streaming version above:
                # a later round can time out (e.g. a large web-search
                # tool result makes the next request too slow for the
                # provider). If we already have at least one round of
                # real tool results, don't lose that data - fall back
                # to answering with what was already gathered instead
                # of failing the whole response. If this is the very
                # first round (nothing gathered yet), preserve the
                # original behavior and let the outer except handle
                # it.
                # ----------------------------------------------------

                if all_tool_calls:

                    logger.exception(
                        "STREAM TOOL ROUND FAILED (iteration=%s), "
                        "falling back to results gathered so far: "
                        "conversation_id=%s",
                        _iteration,
                        conversation_id,
                    )

                    round_failed = True

                else:

                    raise

            if round_failed:
                break

            tool_calls = _parse_tool_calls(
                tool_call_chunks
            )

            # ====================================================
            # NO (MORE) TOOLS THIS ROUND
            #
            # Two different situations land here, and they must be
            # handled differently:
            #
            #   1. No tool was ever called this whole turn (this is
            #      round 0 and all_tool_calls is still empty) ->
            #      there is nothing to reason over, this round's
            #      streamed content IS the final answer. Stream it
            #      straight through, exactly like the original
            #      single-round NO TOOLS case.
            #
            #   2. Tool(s) WERE called in an earlier round this turn
            #      (all_tool_calls is non-empty), and the model has
            #      now stopped calling tools and started producing a
            #      plain-text reply based on those results. THIS
            #      round's raw streamed text must be discarded (not
            #      shown to the user) and control must fall through
            #      to the REASONING block below, so the retrieved
            #      tool data is actually synthesized by the
            #      reasoning model and the "thinking" stage events
            #      are emitted. Previously this branch incorrectly
            #      did `return` here for BOTH situations, which
            #      skipped the reasoning block entirely any time
            #      tools were used - that was the bug causing
            #      reasoning to never show up.
            # ====================================================

            if not tool_calls:

                if not all_tool_calls:

                    for chunk in streamed_chunks:

                        content = getattr(
                            chunk,
                            "content",
                            None,
                        )

                        if content:

                            yield {
                                "type": "answer",
                                "content": content,
                            }

                    return

                break

            # ====================================================
            # TOOLS DETECTED THIS ROUND
            # ====================================================

            logger.info(
                "STREAM TOOL CALLS: tools=%s "
                "conversation_id=%s",
                [
                    tool_call["name"]
                    for tool_call in tool_calls
                ],
                conversation_id,
            )

            all_tool_calls.extend(
                tool_calls
            )

            # ====================================================
            # RECONSTRUCT AI TOOL RESPONSE FOR THIS ROUND
            # ====================================================

            full_ai_response = None

            for chunk in streamed_chunks:

                if full_ai_response is None:

                    full_ai_response = chunk

                else:

                    full_ai_response = (
                        full_ai_response + chunk
                    )

            if full_ai_response is None:

                raise RuntimeError(
                    "Unable to reconstruct "
                    "tool-call response"
                )

            round_messages = [
                full_ai_response
            ]

            # ====================================================
            # EXECUTE TOOLS FOR THIS ROUND
            # ====================================================

            (
                extra_messages,
                collected_images,
                location_request,
                map_location,
            ) = _execute_tool_calls(
                tools=tools,
                tool_calls=tool_calls,
                conversation_id=conversation_id,
                log_prefix="STREAM ",
                known_latitude=latitude,
                known_longitude=longitude,
            )

            round_messages.extend(
                extra_messages
            )

            if images_output is not None:

                images_output.extend(
                    collected_images
                )

            # ====================================================
            # GENERATED IMAGE EVENT
            # ====================================================

            for image in collected_images:
                if (
                    image.get("type")
                    == "generated"
                    and image.get("url")
                ):
                    yield {
                        "type": "generated_image",
                        "url": image.get(
                            "url"
                        ),
                        "filename": image.get(
                            "filename"
                        ),
                        "prompt": image.get(
                            "prompt"
                        ),
                        "model": image.get(
                            "model"
                        ),
                    }

            # ====================================================
            # LOCATION REQUEST -> stop immediately, same as before
            # ====================================================

            if location_request is not None:

                logger.info(
                    "LOCATION REQUESTED: "
                    "conversation_id=%s",
                    conversation_id,
                )

                yield {
                    "type": "location_request",
                    "content": (
                        "I need your location to help "
                        "with that. Please share it "
                        "using the location picker."
                    ),
                    "methods": location_request.get(
                        "methods",
                        [
                            "current_location",
                            "search",
                            "map",
                        ],
                    ),
                }

                return

            # ====================================================
            # MAP LOCATION -> emit event, then continue looping so
            # a follow-up tool (e.g. get_distance_bw_2_locations)
            # can still run using the coordinates just resolved.
            # ====================================================

            if map_location is not None:

                logger.info(
                    "MAP LOCATION FOUND: "
                    "conversation_id=%s",
                    conversation_id,
                )

                yield {
                    "type": "map_location",
                    "content": (
                        f"Showing "
                        f"{map_location.get('name')} "
                        "on the map."
                    ),
                    "latitude": map_location.get(
                        "latitude"
                    ),
                    "longitude": map_location.get(
                        "longitude"
                    ),
                    "name": map_location.get(
                        "name"
                    ),
                    "address": map_location.get(
                        "address"
                    ),
                }

            all_tool_messages.extend(
                round_messages
            )

            current_messages = (
                current_messages + round_messages
            )

        # ========================================================
        # REASONING
        # ========================================================

        use_reasoning = _should_use_reasoning(
            tool_calls=all_tool_calls,
            collected_images=[],
            extra_messages=all_tool_messages,
            conversation_id=conversation_id,
        )

        logger.info(
            "REASONING %s: conversation_id=%s",
            "SELECTED"
            if use_reasoning
            else "SKIPPED",
            conversation_id,
        )

        if use_reasoning:

            reasoning_answer_yielded = False

            try:

                logger.info(
                    "REASONING START: model=%s "
                    "conversation_id=%s",
                    REASONING_MODEL_NAME,
                    conversation_id,
                )

                # ----------------------------------------------------
                # MULTI-STAGE THINKING STATUS
                #
                # _stream_with_thinking_split() below never actually
                # yields a "thinking"-type piece (it only discards
                # raw <think> content for safety and yields "answer"
                # pieces) — so previously the ONLY "thinking" event
                # the UI ever saw was the single hardcoded line
                # emitted here, once, no matter how long reasoning
                # took. That's why the UI showed one static line
                # instead of a proper multi-step process.
                #
                # These are still not raw chain-of-thought — just a
                # short sequence of human-readable stage labels
                # driven by which tools actually ran this turn, so
                # the UI gets a believable step-by-step trail again
                # (Understanding -> Reviewing <tool(s)> -> Analyzing
                # -> Preparing answer) without exposing model
                # internals.
                # ----------------------------------------------------

                yield {
                    "type": "thinking",
                    "content": "Understanding your question...",
                }

                stage_message = _build_stage_message(
                    all_tool_calls
                )

                if stage_message:

                    yield {
                        "type": "thinking",
                        "content": stage_message,
                    }

                yield {
                    "type": "thinking",
                    "content": (
                        "Analyzing the "
                        "retrieved information..."
                    ),
                }

                reasoning_messages = (
                    _build_reasoning_messages(
                        base_messages=messages,
                        tool_messages=all_tool_messages,
                    )
                )

                answer_stage_sent = False

                for piece in (
                    _stream_with_thinking_split(
                        reasoning_llm.stream(
                            reasoning_messages
                        )
                    )
                ):

                    if not piece.get(
                        "content"
                    ):

                        continue

                    if (
                        piece["type"]
                        == "thinking"
                    ):

                        # ------------------------------------------------
                        # Safe status only.
                        # Do NOT expose raw reasoning.
                        # ------------------------------------------------

                        yield {
                            "type": "thinking",
                            "content": (
                                "Analyzing the "
                                "retrieved information..."
                            ),
                        }

                    elif (
                        piece["type"]
                        == "answer"
                    ):

                        if not answer_stage_sent:

                            yield {
                                "type": "thinking",
                                "content": (
                                    "Preparing your answer..."
                                ),
                            }

                            answer_stage_sent = True

                        reasoning_answer_yielded = True

                        yield {
                            "type": "answer",
                            "content": piece[
                                "content"
                            ],
                        }

                if reasoning_answer_yielded:

                    logger.info(
                        "REASONING COMPLETE: "
                        "conversation_id=%s",
                        conversation_id,
                    )

                    return

                logger.warning(
                    "REASONING EMPTY RESULT: "
                    "conversation_id=%s",
                    conversation_id,
                )

            except Exception:

                logger.exception(
                    "REASONING FAILED: "
                    "conversation_id=%s",
                    conversation_id,
                )

        # ========================================================
        # MAIN LLM FALLBACK
        #
        # IMPORTANT:
        # No tools are bound here.
        # ========================================================

        final_messages = (
            messages + all_tool_messages
        )

        for chunk in llm.stream(
            final_messages
        ):

            content = getattr(
                chunk,
                "content",
                None,
            )

            if content:

                yield {
                    "type": "answer",
                    "content": content,
                }

    except HTTPException:

        raise

    except Exception as exc:

        logger.exception(
            "LLM STREAM RESPONSE ERROR: "
            "conversation_id=%s error=%s",
            conversation_id,
            str(exc),
        )

        raise RuntimeError(
            "Unable to generate streaming response"
        ) from exc


# ============================================================
# GENERATE TITLE
# ============================================================


def generate_title(
    question: str,
) -> str:

    try:

        title_prompt = (
            ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """
Generate a short, clear title
(3-6 words) for a conversation
that starts with the given user message.

Rules:
- Do not use quotes.
- Do not add punctuation at the end.
- Return only the title.
""",
                    ),
                    (
                        "human",
                        "{question}",
                    ),
                ]
            )
        )

        messages = (
            title_prompt.format_messages(
                question=question
            )
        )

        response = llm.invoke(
            messages
        )

        title = (
            response.content.strip()
        )

        return (
            title[:100]
            if title
            else question[:50]
        )

    except Exception as exc:

        logger.exception(
            "TITLE GENERATION ERROR: error=%s",
            str(exc),
        )

        return question[:50]


# ============================================================
# SUGGESTIONS
# ============================================================


SUGGESTIONS_MAX_COUNT = int(
    os.getenv(
        "SUGGESTIONS_MAX_COUNT",
        "3",
    )
)


def generate_suggestions(
    question: str,
    answer: str,
    chat_history: Optional[list[dict]] = None,
) -> list[str]:

    try:

        chat_history = (
            chat_history or []
        )

        recent_history = (
            chat_history[-6:]
        )

        history_text = "\n".join(
            f"{message.get('role', '')}: "
            f"{message.get('content', '')}"
            for message in recent_history
        )

        if not history_text:

            history_text = (
                "No previous conversation history."
            )

        suggestions_prompt = (
            ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        f"""
You suggest short follow-up questions
a user might naturally want to ask next.

Rules:
- Suggest at most {SUGGESTIONS_MAX_COUNT}
  follow-up questions.
- Each suggestion must be a question.
- Keep each suggestion short.
- Suggestions must be directly relevant.
- Do not repeat the current question.
- Do not use numbering.
- Do not use bullet points.
- Return exactly one suggestion per line.
- Return nothing if no useful suggestions exist.
""",
                    ),
                    (
                        "human",
                        """
Recent Conversation History:

{history}

Question Just Asked:

{question}

Answer Just Given:

{answer}
""",
                    ),
                ]
            )
        )

        messages = (
            suggestions_prompt.format_messages(
                history=history_text,
                question=question,
                answer=answer,
            )
        )

        response = llm.invoke(
            messages
        )

        raw_text = (
            response.content or ""
        ).strip()

        if not raw_text:

            return []

        suggestions = []

        for line in raw_text.splitlines():

            cleaned = line.strip(
                " \t-*•\"'"
            )

            if not cleaned:
                continue

            suggestions.append(
                cleaned
            )

            if (
                len(suggestions)
                >= SUGGESTIONS_MAX_COUNT
            ):

                break

        return suggestions

    except Exception as exc:

        logger.exception(
            "SUGGESTIONS GENERATION ERROR: "
            "error=%s",
            str(exc),
        )

        return []