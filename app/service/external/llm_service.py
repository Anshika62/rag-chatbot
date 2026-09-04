import json
import logging
import os
from typing import Generator, Optional
from openai import OpenAI
from fastapi import HTTPException, status
from langchain_core.prompts import ChatPromptTemplate
from langchain_openrouter import ChatOpenRouter
import requests

from app.service.tools.conversation_tool import (
    create_conversation_tools,
)

logger = logging.getLogger(__name__)


# ============================================================
# LLM CONFIGURATION
# ============================================================
#
# Both models are configurable via environment variables instead
# of being hard-coded, per the "environment variables for
# model/API configuration" requirement. Defaults preserve the
# existing behavior for the main LLM.
#
# MAIN LLM (llm):
#   Handles normal conversation + decides which tool(s) to call.
#   Used for everything by default.
#
# REASONING LLM (reasoning_llm):
#   A separate, reasoning-capable model used ONLY for the final
#   answer-synthesis step, and only when the tool results are
#   non-trivial (multiple tools, multiple retrieved chunks, or
#   text+image combined — see _should_use_reasoning below).
#   Simple chat ("Hello", "What is Python?") and simple single-
#   tool calls (weather, datetime, a single short KB hit) never
#   reach this model, so they stay fast and cheap.
#
#   The reasoning model is qwen/qwen3.6-27b, currently Groq's
#   highest-intelligence-ranked reasoning model. Unlike
#   openai/gpt-oss-120b (which rejects reasoning_format entirely
#   and only exposes reasoning via a separate include_reasoning
#   flag), qwen3.6-27b officially supports reasoning_format="raw",
#   which makes Groq inline the chain-of-thought as
#   <think>...</think> at the start of the streamed content —
#   exactly what _stream_with_thinking_split() below is built to
#   split into "thinking" vs "answer" pieces. That function also
#   still checks additional_kwargs["reasoning_content"] as a second
#   mechanism, so switching REASONING_MODEL to a gpt-oss model
#   later would keep working without further changes. Every piece
#   — thinking or answer — is yielded as {"type": "thinking"/
#   "answer", "content": ...} so the frontend can render a live
#   "thinking..." trace as it happens, while still saving only the
#   "answer" portion as the final message content.
#
#   The raw chain-of-thought is streamed to the user as-is (no
#   artificial shortening/filtering of "thinking" content).
# ============================================================


LLM_MODEL_NAME = os.getenv(
    "LLM_MODEL",
    "openai/gpt-oss-20b",
)

# Reasoning model runs on Cloudflare Workers AI. Default is
# @cf/deepseek-ai/deepseek-r1-distill-qwen-32b, Cloudflare's currently supported free/
# freemium "Reasoning"-tagged model for this use case. It uses
# the exact same REST contract as the previous
# deepseek-r1-distill-qwen-32b default (messages in, "response"
# field per SSE line out) and still emits its chain-of-thought
# as inline <think>...</think> tags, so _stream_with_thinking_
# split() below works completely unchanged.

OPENROUTER_API_KEY = (
    os.getenv("OPENROUTER_API_KEY") or ""
).strip()


REASONING_MODEL_NAME = os.getenv(
    "REASONING_MODEL",
    "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b",
)

GROQ_API_KEY = (os.getenv("GROQ_API_KEY") or "").strip()

CLOUDFLARE_ACCOUNT_ID = (os.getenv("CLOUDFLARE_ACCOUNT_ID") or "").strip()
CLOUDFLARE_API_TOKEN = (os.getenv("CLOUDFLARE_API_TOKEN") or "").strip()

CLOUDFLARE_BASE_URL = "https://api.cloudflare.com/client/v4/accounts"


llm = ChatOpenRouter(
    model=LLM_MODEL_NAME,
    temperature=0.2,
    api_key=OPENROUTER_API_KEY,
    max_retries=2,
)


# ============================================================
# CLOUDFLARE REASONING ADAPTER
#
# langchain_community's CloudflareWorkersAI wrapper has two
# problems for our use case:
#
#   1. It's a completion-style LLM (plain string "prompt"), not
#      a chat model — Cloudflare's reasoning/chat models actually
#      expect a "messages" array, same shape as OpenAI's chat
#      format.
#   2. Its built-in _stream() method has a parsing bug: it slices
#      the first 6 characters off every non-empty SSE line and
#      tries to json.loads() it, without checking the line
#      actually starts with "data: " first. Any other line (an
#      SSE comment, a differently-shaped event, or an error body
#      that isn't in "data: ..." format at all) makes it crash
#      with a JSONDecodeError instead of just skipping that line.
#
# This adapter bypasses that library entirely and talks to the
# Cloudflare REST API directly with `requests`, using the proper
# "messages" format and a defensive line-by-line SSE parser that
# silently skips anything that isn't a valid "data: {...}" line
# instead of raising.
#
# It exposes only `.invoke(messages)` and `.stream(messages)`,
# since those are the only two methods called on reasoning_llm
# anywhere else in this file — so nothing else needs to change.
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
            role = getattr(message, "type", "human")
            content = getattr(message, "content", str(message))

        cf_messages.append(
            {
                "role": _ROLE_MAP.get(role, "user"),
                "content": content,
            }
        )

    return cf_messages


class _ReasoningChunk:

    __slots__ = ("content", "additional_kwargs")

    def __init__(self, content: str):
        self.content = content
        self.additional_kwargs = {}


class _ReasoningResponse:

    def __init__(self, content: str):
        self.content = content


REASONING_MAX_TOKENS = int(
    os.getenv(
        "REASONING_MAX_TOKENS",
        "4096",
    )
)


class _CloudflareReasoningLLM:

    def __init__(self, account_id: str, api_token: str, model: str):

        self._account_id = account_id
        self._api_token = api_token
        self._model = model
        self._endpoint = (
            f"{CLOUDFLARE_BASE_URL}/{account_id}/ai/run/{model}"
        )

    def _request(self, messages, stream: bool):

        headers = {
            "Authorization": f"Bearer {self._api_token}",
            "Content-Type": "application/json",
        }

        payload = {
            "messages": _reasoning_messages_to_cf_messages(messages),
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
                f"Cloudflare Workers AI request failed "
                f"(status={response.status_code}): {response.text}"
            )

        return response

    def invoke(self, messages):

        response = self._request(messages, stream=False)

        data = response.json()

        text = data.get("result", {}).get("response", "")

        return _ReasoningResponse(text)

    def stream(self, messages):

        response = self._request(messages, stream=True)

        for raw_line in response.iter_lines():

            if not raw_line:
                continue

            if not raw_line.startswith(b"data: "):
                # SSE comment lines, differently-shaped events,
                # or anything else that isn't a data payload —
                # skip instead of crashing.
                continue

            payload_bytes = raw_line[len(b"data: "):]

            if payload_bytes.strip() == b"[DONE]":
                break

            try:

                data = json.loads(payload_bytes)

            except json.JSONDecodeError:

                logger.warning(
                    "REASONING STREAM: skipping unparseable "
                    "line: %r",
                    raw_line,
                )

                continue

            token = data.get("response")

            if token:

                yield _ReasoningChunk(token)


reasoning_llm = _CloudflareReasoningLLM(
    account_id=CLOUDFLARE_ACCOUNT_ID,
    api_token=CLOUDFLARE_API_TOKEN,
    model=REASONING_MODEL_NAME,
)


THINK_START_TAG = "<think>"
THINK_END_TAG = "</think>"


def _is_image_content_type(content_type) -> bool:
    """
    True for either shape "content_type" appears in across the
    tool results this function consumes:
      - the bare Qdrant payload discriminator "image" (what
        semantic-search hits from search_knowledge_base carry)
      - a real mime type like "image/png" (what direct image
        retrieval — content_type="image" branch in search_kb.py —
        carries, taken from image_doc.mime_type)

    A plain `.startswith("image/")` check only matches the second
    shape and silently misses every semantic-search image hit,
    so those never made it into collected_images.
    """

    if not content_type:
        return False

    content_type = str(content_type)

    return (
        content_type == "image"
        or content_type.startswith("image/")
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
4. Document image analysis tool (analyze_document_image) — for
   analyzing a SPECIFIC image already extracted from an uploaded
   document/PDF, identified by document_id
5. Current date and time tool
6. Weather tool
7. Get user location tool (get_location) — for requesting the
   user's OWN current location when it is required and not
   already known
8. Search nearby places tool (search_nearby_places) — for
   places/points-of-interest search once real coordinates are
   known
9. Image analysis tool (analyze_image) — only present when the
   user has attached an image directly to their CURRENT message
10. Web search tool (tavily_web_search) — for searching the live/public
    web for information that is not reliably available from the
    conversation or uploaded knowledge base
11. Find location on map tool (find_location_on_map) — for
    locating ONE specific named place so it can be shown as a
    pin on a map
Rules:

- Answer normal conversational questions directly.

- Use get_conversation_history when the provided history is
  insufficient.

- Use search_knowledge_base whenever the answer may be present
  in uploaded documents or the knowledge base.

- If the user asks about an uploaded document, PDF, file, policy,
  manual, FAQ, guideline, documentation, or indexed content,
  search the knowledge base before answering.

- If an uploaded document is available for the current
  conversation, ALWAYS use search_knowledge_base first for
  factual questions that could reasonably be answered from
  that document.

- The user does NOT need to mention the document, PDF, file,
  or say "in this uploaded document".

- For example, if an uploaded document is available and the
  user asks "Who is Anshika?", "What is Anshika's role?", or
  "When did Anshika join?", search_knowledge_base before
  answering.

- When an uploaded document is available, do not answer a
  factual question from general knowledge if the uploaded
  document could contain the answer.

- When search_knowledge_base returns relevant document content,
  use that retrieved content as the source of truth.

- Do not invent information from uploaded documents.

- If an image was attached directly to the current message
  (the analyze_image tool is available), use analyze_image to
  answer questions about THAT image.

- Prefer analyze_image over search_knowledge_base for a
  just-attached image.

- Only use search_knowledge_base for images that were uploaded
  previously as part of the document knowledge base.

- When search_knowledge_base returns an image result, use its
  caption/text to answer the question and treat the returned
  image metadata as the related image.

- Do not confuse an image document_id with its parent PDF
  document_id.

- If the user asks about an image inside a PDF, use the image
  result whose parent_document_id matches the PDF.

- When the user asks you to explain, interpret, or describe what
  an already-uploaded image/diagram/chart/figure actually shows
  (not just "is there an image"), first use search_knowledge_base
  (content_type="image") to find the relevant image and its
  document_id, then call analyze_document_image with that
  document_id and the user's specific question. Do not answer
  such questions using only the cached caption/OCR text if
  analyze_document_image is available — the actual image should
  be analyzed for the user's specific question.

- Do not guess a document_id for analyze_document_image. Only use
  a document_id that was actually returned by search_knowledge_base
  in this conversation.

- If the knowledge-base search finds no relevant information,
  say that the information was not found in the uploaded
  knowledge base.

- Use conversation history for follow-up questions.

- Keep answers clear and concise.

- Use get_current_datetime when the user asks for the current
  date or time.

- Never guess the current date or time. Always use
  get_current_datetime for current date/time questions.

- Use get_weather when the user asks about current weather,
  temperature, rainfall, humidity, wind, or weather conditions
  for a location.

- Never invent current weather information. Always use
  get_weather for current weather questions.

- Use tavily_web_search when the user asks for information that
  requires live/current web information, recent public information,
  web research, online sources, current events, newly published
  information, or information that is not available in the
  conversation or uploaded knowledge base.

- tavily_web_search is a GENERAL web-search tool. Do not restrict it
  to location or places. It can be used for current events, recent
  information, public websites, articles, documentation, research,
  comparisons, and other questions that benefit from web search.

- Do not use tavily_web_search when the answer is clearly available
  from the uploaded knowledge base or normal conversation context,
  unless the user also explicitly needs current/external web
  information.

- Do not invent web-search results, URLs, facts, or source details.
  Base web-researched claims on the information returned by
  tavily_web_search.

- When web search results are returned together with knowledge-base
  results or other tool results, compare and synthesize them
  carefully. Prefer the source that is most directly relevant to the
  user's question and do not treat unrelated retrieved results as
  authoritative.

- Use get_location ONLY when the user's own current/live
  location is required to answer (e.g. "near me", "closest to
  me", "here") and it has not already been provided in the
  current question or the conversation history.

- Do NOT use get_location when the user names a specific place
  (city, address, landmark) — use get_weather or
  search_knowledge_base as appropriate instead.

- Never guess, assume, or invent the user's location. If
  get_location has been called, wait for the user to actually
  provide it in a later message rather than answering as if a
  location is already known.

- Always check the "User Location" section below (in the human
  message) before deciding whether to call get_location. If it
  states that the user's location is already known, do NOT call
  get_location — use those exact coordinates directly when
  calling search_nearby_places, get_distance_bw_2_locations, or
  compare_travel_modes.

- Use find_location_on_map when the user names ONE specific
  place (not a category) and wants to see it located on a map —
  e.g. "show me Vijay Nagar on the map", "where is Bargi Dam".
  Do not use this for the user's own current location (use
  get_location instead), and do not use it for finding multiple
  places of a category nearby (use search_nearby_places instead).

- After calling find_location_on_map, do NOT invent or embed any
  map image URL, static-map link, or third-party maps deep link
  (Google Maps, Yandex Maps, etc.) in your answer — the frontend
  renders the actual map separately using the tool's coordinates.
  Just confirm the place was found in plain text (e.g. "Here's
  Vijay Nagar, Indore — you can see it on the map above.").

- When search_knowledge_base or analyze_document_image returns an
  image (a result whose content_type starts with "image/", or a
  document_id returned by analyze_document_image), and that image
  helps answer the user's question, embed it INLINE in your
  answer text, exactly at the point where it is relevant to what
  you are explaining — do not describe the image and then list it
  separately, and do not collect images to mention only at the
  end of your answer.

- To embed an image inline, use this exact markdown image syntax,
  using the "url" field from the tool result:
  ![short description](url)

- Only use a document_id/url that was actually returned by
  search_knowledge_base or analyze_document_image earlier in this
  same turn. Never invent, guess, or reuse a document_id/url from
  a previous conversation turn.

- When tavily_web_search returns web results, use those results as
  the source of truth for the web-researched portion of the answer.
  Do not invent URLs, publication details, source names, or facts
  that are not supported by the returned results.

- If both web-search results and uploaded-document results are
  available, keep the two sources distinct and synthesize them
  according to the user's question. Do not replace uploaded
  document facts with web information unless the question requires
  current/external information.

- If more than one image is relevant to different parts of your
  answer, place each image inline right next to the text it
  relates to, not grouped together in one place.
"""


# ============================================================
# BUILD LOCATION CONTEXT
#
# Turns the latitude/longitude/address (if any) that arrived
# with THIS request into a plain-language line the LLM can read
# in the human message. This is what lets the agent skip calling
# get_location on every subsequent turn once the frontend has
# already sent real coordinates — see SYSTEM_PROMPT rule above.
# ============================================================


def _build_location_context(
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    address: Optional[str] = None,
) -> str:

    if latitude is not None and longitude is not None:

        location_line = (
            f"The user's current location is ALREADY KNOWN: "
            f"latitude={latitude}, longitude={longitude}"
        )

        if address:

            location_line += f", address='{address}'"

        location_line += (
            ". Do NOT call get_location — this location is "
            "already available. Use these exact coordinates "
            "directly when calling search_nearby_places, "
            "get_distance_bw_2_locations, or compare_travel_modes."
        )

        return location_line

    return (
        "The user's current location is NOT known yet. If the "
        "current question requires it (e.g. 'near me', "
        "'closest to me', 'here'), call get_location first."
    )


# ============================================================
# BUILD LLM MESSAGES
# ============================================================


def _build_messages(
    question: str,
    chat_history: Optional[list[dict]] = None,
    document_available: bool = False,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    address: Optional[str] = None,
):

    chat_history = chat_history or []

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

    document_context = (
        "YES. This turn is scoped to one specific uploaded "
        "document. If the current question could be answered "
        "from it, use search_knowledge_base before answering."
        if document_available
        else
        "Documents MAY be available in the knowledge base for "
        "this user — either global documents (uploaded without "
        "being tied to a conversation) or documents uploaded "
        "within this conversation. No single document_id is "
        "pre-selected for this turn, but search_knowledge_base "
        "still searches across all documents accessible to this "
        "user/conversation. If the current question could "
        "reasonably be answered from an uploaded document, use "
        "search_knowledge_base before answering from general "
        "knowledge — do not assume no document exists just "
        "because none is pre-selected."
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
                """Conversation History:

{history}

Uploaded Document Available:

{document_context}

User Location:

{location_context}

Current User Question:

{question}""",
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
# BIND TOOLS
# ============================================================


def _bind_tools(tools: list):

    return (
        llm.bind_tools(tools)
        if tools
        else llm
    )


# ============================================================
# GET TOOL
# ============================================================


def _get_tool(
    tools: list,
    tool_name: str,
):

    return next(
        (
            current_tool
            for current_tool in tools
            if current_tool.name == tool_name
        ),
        None,
    )


# ============================================================
# EXECUTE TOOL CALLS
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
        # SAFETY NET — LOCATION ALREADY KNOWN
        #
        # If the frontend already sent real coordinates for
        # this turn but the LLM called get_location anyway
        # (e.g. it ignored the "User Location" context), don't
        # re-trigger the frontend location picker. Short-circuit
        # with the already-known coordinates instead, so the
        # agent can immediately continue with
        # search_nearby_places / get_distance_bw_2_locations /
        # compare_travel_modes in its next step.
        # ====================================================

        if (
            tool_name == "get_location"
            and known_latitude is not None
            and known_longitude is not None
        ):

            logger.info(
                "%sLOCATION ALREADY KNOWN, SKIPPING "
                "get_location: conversation_id=%s",
                log_prefix,
                conversation_id,
            )

            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": (
                        f"Location already known: "
                        f"latitude={known_latitude}, "
                        f"longitude={known_longitude}. Use "
                        "this directly, do not ask the user "
                        "again."
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
                "%sTOOL FAILED: tool=%s args=%s "
                "conversation_id=%s",
                log_prefix,
                tool_name,
                tool_args,
                conversation_id,
            )

            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": (
                        f"The '{tool_name}' tool failed and "
                        "is temporarily unavailable. Let the "
                        "user know and answer with whatever "
                        "other information is available."
                    ),
                }
            )

            continue

        logger.info(
            "%sTOOL RESULT: tool=%s conversation_id=%s",
            log_prefix,
            tool_name,
            conversation_id,
        )

        # ====================================================
        # COLLECT IMAGE REFERENCES
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
                    _is_image_content_type(content_type)
                    and item.get("document_id")
                ):

                    image_document_id = str(
                        item.get("document_id")
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
                tool_result.get("document_id")
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
        # DETECT LOCATION REQUEST
        #
        # get_location never returns real coordinates — it
        # returns this marker dict to signal that the frontend
        # should be told (via a dedicated SSE event) to show a
        # location-selection UI. See location_tool.py.
        # ====================================================

        elif (
            tool_name == "get_location"
            and isinstance(tool_result, dict)
            and tool_result.get("action")
            == "request_location"
        ):

            location_request = tool_result

        # ====================================================
        # DETECT MAP LOCATION
        #
        # find_location_on_map returns real coordinates for a
        # single named place, marked with action="show_map".
        # Captured here so generate_answer_stream can emit a
        # dedicated "map_location" piece for the frontend to
        # render a pin on a map. See geocode_tool.py.
        # ====================================================

        elif (
            tool_name == "find_location_on_map"
            and isinstance(tool_result, dict)
            and tool_result.get("success")
            and tool_result.get("action") == "show_map"
        ):

            map_location = tool_result

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
#
# UPDATED: reasoning is now used for EVERY turn where at least
# one tool was called, regardless of which tool(s) or how much
# text they returned. This fixes two problems seen in prod:
#
#   1. Reasoning showing up inconsistently — tools like
#      find_location_on_map (map pin) or search_nearby_places
#      (fuel/station/etc.) were never in the old allow-list, so
#      those turns silently skipped reasoning and went straight
#      to the (buggy) tool-bound final-answer path.
#   2. reasoning_llm never has tools bound to it, so routing
#      every tool-based turn through it also means it can never
#      emit an empty tool-call chunk instead of text — removing
#      the "I was unable to generate a response" failure mode
#      for those turns as a side effect.
#
# Turns with NO tool calls (plain chit-chat, "hi", etc.) still
# skip reasoning and answer directly from the Main LLM, so
# simple conversation stays fast/cheap.
# ============================================================


def _should_use_reasoning(
    tool_calls: list,
    collected_images: list,
    extra_messages: list,
) -> bool:

    return bool(tool_calls)


# ============================================================
# THINKING/ANSWER STREAM SPLITTER
#
# The reasoning model emits a single continuous token stream that
# looks like:
#
#     <think> ...chain of thought... </think> ...final answer...
#
# This splits that stream into separate "thinking" and "answer"
# events as it arrives, so the caller can show live "thinking..."
# output (like other reasoning-model chat UIs) without it ending
# up as part of the saved/displayed final answer. A small tail of
# text is always held back while scanning for a tag, in case a
# tag like "<think>" is split across two streamed chunks.
#
# Two different mechanisms are checked on every chunk, since
# different reasoning models expose their chain-of-thought
# differently:
#
#   1. additional_kwargs["reasoning_content"] (or ["reasoning"])
#      — used by openai/gpt-oss-20b / openai/gpt-oss-120b on Groq,
#      which return reasoning as a separate field per chunk
#      instead of inlining it in .content.
#
#   2. <think>...</think> tags inline inside .content — used by
#      DeepSeek-R1-distill and some other reasoning models.
#
# A model only ever uses one of the two, so only one branch will
# ever produce output for a given REASONING_MODEL — the other is
# simply a silent no-op.
# ============================================================


def _stream_with_thinking_split(
    model_stream,
):

    state = "answer"
    pending = ""

    for chunk in model_stream:

        # ---- mechanism 1: separate reasoning field per chunk ----

        extra_kwargs = getattr(
            chunk,
            "additional_kwargs",
            None,
        ) or {}

        reasoning_piece = (
            extra_kwargs.get("reasoning_content")
            or extra_kwargs.get("reasoning")
        )

        if reasoning_piece:

            yield {
                "type": "thinking",
                "content": reasoning_piece,
            }

        # ---- mechanism 2: inline <think> tags inside .content ----

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

                tag_index = pending.find(
                    THINK_START_TAG
                )

                if tag_index == -1:

                    safe_length = max(
                        0,
                        len(pending)
                        - len(THINK_START_TAG),
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

                if tag_index:

                    yield {
                        "type": "answer",
                        "content": pending[
                            :tag_index
                        ],
                    }

                pending = pending[
                    tag_index
                    + len(THINK_START_TAG):
                ]

                state = "thinking"

            else:

                tag_index = pending.find(
                    THINK_END_TAG
                )

                if tag_index == -1:

                    safe_length = max(
                        0,
                        len(pending)
                        - len(THINK_END_TAG),
                    )

                    if safe_length:

                        yield {
                            "type": "thinking",
                            "content": pending[
                                :safe_length
                            ],
                        }

                        pending = pending[
                            safe_length:
                        ]

                    break

                if tag_index:

                    yield {
                        "type": "thinking",
                        "content": pending[
                            :tag_index
                        ],
                    }

                pending = pending[
                    tag_index
                    + len(THINK_END_TAG):
                ]

                state = "answer"

    if pending:

        yield {
            "type": state,
            "content": pending,
        }


# ============================================================
# STRIP THINKING (non-streamed reasoning responses)
#
# Mirrors _stream_with_thinking_split above, but for a single
# already-complete response string (used by generate_answer,
# the non-streaming path). Removes any <think>...</think> block
# so the model's private chain-of-thought is never returned as
# part of the final answer.
# ============================================================


def _strip_thinking(text: str) -> str:

    if not text:
        return text

    result = []
    remaining = text

    while True:

        start_index = remaining.find(THINK_START_TAG)

        if start_index == -1:
            result.append(remaining)
            break

        result.append(remaining[:start_index])

        end_index = remaining.find(
            THINK_END_TAG,
            start_index + len(THINK_START_TAG),
        )

        if end_index == -1:
            # Unclosed tag — drop everything from the tag onward
            # rather than risk leaking a partial chain-of-thought.
            break

        remaining = remaining[
            end_index + len(THINK_END_TAG):
        ]

    return "".join(result).strip()


# ============================================================
# REASONING CONTEXT BUILDER
#
# Builds the minimal set of messages the reasoning model needs:
# the original conversation/question messages plus only the tool
# results actually produced for this turn (retrieved chunks,
# vision analysis, weather/datetime results, etc). We never hand
# the reasoning model the raw database/conversation dump — only
# what _execute_tool_calls already gathered for this turn.
# ============================================================


def _build_reasoning_messages(
    base_messages: list,
    tool_messages: list,
):

    reasoning_instruction = (
        "human",
        "Using ONLY the information above (the question, "
        "conversation context, and tool results), reason "
        "carefully and produce one clear, well-synthesized "
        "final answer for the user.\n\n"
        "Important:\n"
        "- Some retrieved tool results may NOT be relevant to "
        "the user's actual question (retrieval is not perfect). "
        "Silently discard anything irrelevant — do not mention, "
        "summarize, or reference it in your answer.\n"
        "- Only use content that directly helps answer the "
        "question asked.\n"
        "- Write a natural, concise, conversational answer. Do "
        "not dump raw retrieved text, filenames, metadata, or "
        "internal tool details into the answer.\n"
        "- Do not mention that you are a separate reasoning "
        "step, that you used tools, or that some results were "
        "discarded.\n"
        "- If web-search results are present, synthesize only the "
        "relevant information returned by the web-search tool. Do "
        "not invent or assume unsupported web facts or URLs.\n"
        "- If any tool result above includes a retrieved image "
        "(a result whose content_type starts with \"image/\", or "
        "an analyze_document_image result), and that image helps "
        "answer the question, embed it INLINE in your answer, "
        "exactly at the point where it is relevant, using this "
        "exact markdown syntax and the \"url\" field from that "
        "tool result: ![short description](url). Only use a "
        "document_id/url that actually appears in the tool "
        "results above — never invent or guess one. Do not "
        "collect images and place them only at the end.",
    )

    return (
        base_messages
        + tool_messages
        + [reasoning_instruction]
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

    """
    document_id:
        If provided, scopes the knowledge-base search tool
        to that single uploaded document.

    image_paths:
        Local file path(s) of images attached directly to
        THIS message.

    latitude / longitude / address:
        The user's current location, if it was already sent
        with this request (e.g. by the frontend location
        picker). When provided, the LLM is told the location is
        already known so it never re-triggers get_location for
        this turn.
    """

    try:

        messages = _build_messages(
            question=question,
            chat_history=chat_history,
            document_available=bool(document_id),
            latitude=latitude,
            longitude=longitude,
            address=address,
        )

        tools = _create_tools(
            db=db,
            user_id=user_id,
            conversation_id=conversation_id,
            document_id=document_id,
            image_paths=image_paths,
        )

        llm_with_tools = _bind_tools(tools)

        response = llm_with_tools.invoke(
            messages
        )

        if not response.tool_calls:

            return response.content

        logger.info(
            "LLM TOOL CALLS: tools=%s "
            "conversation_id=%s",
            [
                tool_call["name"]
                for tool_call in response.tool_calls
            ],
            conversation_id,
        )

        tool_messages = [
            response
        ]

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

        tool_messages.extend(
            extra_messages
        )

        if images_output is not None:

            images_output.extend(
                collected_images
            )

        # ====================================================
        # REASONING STAGE (final-answer synthesis)
        #
        # Now runs for every turn that made at least one tool
        # call (see _should_use_reasoning above).
        # ====================================================

        use_reasoning = _should_use_reasoning(
            tool_calls=response.tool_calls,
            collected_images=collected_images,
            extra_messages=extra_messages,
        )

        logger.info(
            "REASONING %s: conversation_id=%s",
            "SELECTED" if use_reasoning else "SKIPPED",
            conversation_id,
        )

        if use_reasoning:

            try:

                logger.info(
                    "REASONING START: model=%s "
                    "conversation_id=%s",
                    REASONING_MODEL_NAME,
                    conversation_id,
                )

                reasoning_messages = _build_reasoning_messages(
                    base_messages=messages,
                    tool_messages=tool_messages,
                )

                reasoning_response = reasoning_llm.invoke(
                    reasoning_messages
                )

                final_answer = _strip_thinking(
                    reasoning_response.content
                )

                if final_answer:

                    logger.info(
                        "REASONING COMPLETE: conversation_id=%s",
                        conversation_id,
                    )

                    return final_answer

                logger.warning(
                    "REASONING EMPTY RESULT, falling back to "
                    "main LLM: conversation_id=%s",
                    conversation_id,
                )

            except Exception:

                logger.exception(
                    "REASONING FAILED, falling back to main "
                    "LLM: conversation_id=%s",
                    conversation_id,
                )

        # ====================================================
        # MAIN LLM FINAL ANSWER (default path / reasoning
        # fallback)
        #
        # UPDATED: uses the plain, tools-UNBOUND `llm` instead
        # of `llm_with_tools`. With tools still bound here, the
        # model could (and did, per prod logs) emit ANOTHER
        # tool_call instead of text when it wasn't fully
        # satisfied with the tool results (e.g. a failed
        # find_location_on_map call, or wanting one more web
        # search) — that tool-call chunk has no .content, so the
        # caller ends up with nothing to show ("I was unable to
        # generate a response"). Using the plain model forces a
        # text answer using whatever tool results are already
        # available.
        # ====================================================

        final_response = llm.invoke(
            messages + tool_messages
        )

        return final_response.content

    except HTTPException:

        raise

    except Exception as exc:

        logger.exception(
            "LLM RESPONSE ERROR: conversation_id=%s "
            "error=%s",
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

        args_text = tool_call["args"].strip()

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

    """
    document_id:
        If provided, scopes the knowledge-base search tool
        to that single uploaded document.

    image_paths:
        Local file path(s) of image(s) attached directly to
        THIS message.

    latitude / longitude / address:
        The user's current location, if it was already sent
        with this request (e.g. by the frontend location
        picker). When provided, the LLM is told the location is
        already known so it never re-triggers get_location for
        this turn.
    """

    try:

        messages = _build_messages(
            question=question,
            chat_history=chat_history,
            document_available=bool(document_id),
            latitude=latitude,
            longitude=longitude,
            address=address,
        )

        tools = _create_tools(
            db=db,
            user_id=user_id,
            conversation_id=conversation_id,
            document_id=document_id,
            image_paths=image_paths,
        )

        llm_with_tools = _bind_tools(tools)

        logger.info(
            "LLM STREAM START: conversation_id=%s "
            "tools=%s",
            conversation_id,
            [
                tool.name
                for tool in tools
            ],
        )

        streamed_chunks = []

        tool_call_chunks = []

        for chunk in llm_with_tools.stream(
            messages
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

        tool_calls = _parse_tool_calls(
            tool_call_chunks
        )

        # ====================================================
        # NORMAL STREAMING RESPONSE
        # ====================================================

        if not tool_calls:

            for chunk in streamed_chunks:

                if chunk.content:

                    yield {
                        "type": "answer",
                        "content": chunk.content,
                    }

            return

        # ====================================================
        # TOOL CALLS DETECTED
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

        # ====================================================
        # RECONSTRUCT AI TOOL-CALL RESPONSE
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
                "Unable to reconstruct tool-call response"
            )

        tool_messages = [
            full_ai_response
        ]

        # ====================================================
        # EXECUTE TOOLS
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

        tool_messages.extend(
            extra_messages
        )

        if images_output is not None:

            images_output.extend(
                collected_images
            )

        # ====================================================
        # LOCATION REQUESTED
        #
        # get_location was called. Do NOT let the LLM synthesize
        # a final answer here — it has no real location and
        # would be forced to guess/hallucinate one. Short-circuit
        # with a single "location_request" piece instead; the
        # streaming caller (rag_service.py) turns this into a
        # dedicated SSE event that tells the frontend to show the
        # location picker.
        #
        # Note: if latitude/longitude were already known for
        # this turn, _execute_tool_calls above short-circuits
        # get_location itself and location_request stays None,
        # so this branch is never hit in that case.
        # ====================================================

        if location_request is not None:

            logger.info(
                "LOCATION REQUESTED: conversation_id=%s",
                conversation_id,
            )

            yield {
                "type": "location_request",
                "content": (
                    "I need your location to help with that. "
                    "Please share it using the location picker."
                ),
                "methods": location_request.get(
                    "methods",
                    ["current_location", "search", "map"],
                ),
            
            }

            return

        # ====================================================
        # MAP LOCATION FOUND
        #
        # find_location_on_map returned real coordinates for a
        # single named place. Unlike get_location, this does NOT
        # short-circuit the answer — the LLM still writes a
        # normal reply using the tool result (see the "do NOT
        # invent map links" SYSTEM_PROMPT rule). This piece just
        # carries the coordinates separately so the frontend can
        # drop a pin on a map alongside the text answer.
        # ====================================================

        if map_location is not None:

            logger.info(
                "MAP LOCATION FOUND: conversation_id=%s",
                conversation_id,
            )

            yield {
                "type": "map_location",
                "content": (
                    f"Showing {map_location.get('name')} on "
                    "the map."
                ),
                "latitude": map_location.get("latitude"),
                "longitude": map_location.get("longitude"),
                "name": map_location.get("name"),
                "address": map_location.get("address"),
            }

        # ====================================================
        # GENERATE FINAL RESPONSE
        # ====================================================

        final_messages = (
            messages + tool_messages
        )

        # ====================================================
        # REASONING STAGE (final-answer synthesis)
        #
        # Now runs for every turn that made at least one tool
        # call (see _should_use_reasoning above).
        # ====================================================

        use_reasoning = _should_use_reasoning(
            tool_calls=tool_calls,
            collected_images=collected_images,
            extra_messages=extra_messages,
        )

        logger.info(
            "REASONING %s: conversation_id=%s",
            "SELECTED" if use_reasoning else "SKIPPED",
            conversation_id,
        )

        if use_reasoning:

            reasoning_yielded_any = False

            try:

                logger.info(
                    "REASONING START: model=%s "
                    "conversation_id=%s",
                    REASONING_MODEL_NAME,
                    conversation_id,
                )

                reasoning_messages = _build_reasoning_messages(
                    base_messages=messages,
                    tool_messages=tool_messages,
                )

                reasoning_answer_yielded = False

                # NOTE: _stream_with_thinking_split's output is
                # yielded as-is — "thinking" pieces are the raw,
                # unfiltered chain-of-thought (no artificial
                # shortening/noise filtering applied), and
                # "answer" pieces pass through unchanged and
                # unbuffered.
                for piece in _stream_with_thinking_split(
                    reasoning_llm.stream(reasoning_messages)
                ):

                    if not piece["content"]:
                        continue

                    if piece["type"] == "thinking":

                        # Live chain-of-thought — shown to the
                        # user as a transient "thinking..." trace.
                        # Never saved as the final answer.

                        yield {
                            "type": "thinking",
                            "content": piece["content"],
                        }

                    elif piece["type"] == "answer":

                        reasoning_yielded_any = True
                        reasoning_answer_yielded = True

                        yield {
                            "type": "answer",
                            "content": piece["content"],
                        }

                if reasoning_answer_yielded:

                    logger.info(
                        "REASONING COMPLETE: conversation_id=%s",
                        conversation_id,
                    )

                    return

                logger.warning(
                    "REASONING EMPTY RESULT, falling back to "
                    "main LLM: conversation_id=%s",
                    conversation_id,
                )

            except Exception:

                logger.exception(
                    "REASONING FAILED: conversation_id=%s",
                    conversation_id,
                )

                # If the reasoning model already streamed part of
                # an answer before failing, do NOT also stream the
                # Main LLM's answer — that would produce a garbled,
                # duplicated response. Only fall back to the Main
                # LLM when reasoning produced nothing at all.
                if reasoning_yielded_any:
                    return

        # ====================================================
        # MAIN LLM FINAL ANSWER (default path / reasoning
        # fallback)
        #
        # UPDATED: streams from the plain, tools-UNBOUND `llm`
        # instead of `llm_with_tools` — see the matching comment
        # in generate_answer() above for why. This is what
        # actually fixes the "I was unable to generate a
        # response" case seen in prod for find_location_on_map /
        # tavily_web_search turns.
        # ====================================================

        for chunk in llm.stream(
            final_messages
        ):

            if chunk.content:

                yield {
                    "type": "answer",
                    "content": chunk.content,
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
            f"Unable to generate streaming response: {exc}"
        ) from exc


# ============================================================
# GENERATE CONVERSATION TITLE
# ============================================================


def generate_title(
    question: str,
) -> str:

    try:

        title_prompt = ChatPromptTemplate.from_messages(
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

        messages = title_prompt.format_messages(
            question=question
        )

        response = llm.invoke(
            messages
        )

        title = response.content.strip()

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
# GENERATE FOLLOW-UP SUGGESTIONS
#
# Called by rag_service.query_documents_stream() AFTER the
# "done" event has already been sent, as a non-critical
# enhancement. Uses the Main LLM (not the reasoning model) to
# propose a small number of short, natural follow-up questions
# the user might want to ask next, based on the question just
# asked and the answer just given (plus recent chat history for
# context).
#
# This intentionally mirrors generate_title()'s structure: a
# small, single-purpose ChatPromptTemplate + llm.invoke() call,
# wrapped in a try/except that swallows all errors and returns
# an empty list on failure, so a broken/slow suggestions call
# can never affect the already-completed answer.
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

    """
    Returns a short list of follow-up-question strings the user
    might want to ask next, based on the just-completed Q&A turn.
    Returns an empty list if generation fails or no good
    suggestions can be produced — callers should treat an empty
    list as "no suggestions" rather than an error.
    """

    try:

        chat_history = chat_history or []

        recent_history = chat_history[-6:]

        history_text = "\n".join(
            f"{message.get('role', '')}: "
            f"{message.get('content', '')}"
            for message in recent_history
        )

        if not history_text:

            history_text = (
                "No previous conversation history."
            )

        suggestions_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"""
You suggest short follow-up questions a user might naturally
want to ask next in a chat conversation, based on the question
they just asked and the answer they just received.

Rules:
- Suggest at most {SUGGESTIONS_MAX_COUNT} follow-up questions.
- Each suggestion must be a short, natural question the USER
  would ask (not a statement, not an instruction to the AI).
- Suggestions must be directly relevant to the question/answer
  above — do not suggest generic or unrelated questions.
- Do not repeat the question that was just asked.
- Do not number the suggestions.
- Do not use quotes or bullet points.
- Return exactly one suggestion per line, and nothing else —
  no preamble, no explanation.
- If no good follow-up questions make sense (e.g. the answer
  already fully resolves the topic, or the turn was just
  small talk), return nothing at all.
""",
                ),
                (
                    "human",
                    """Recent Conversation History:

{history}

Question Just Asked:

{question}

Answer Just Given:

{answer}""",
                ),
            ]
        )

        messages = suggestions_prompt.format_messages(
            history=history_text,
            question=question,
            answer=answer,
        )

        response = llm.invoke(
            messages
        )

        raw_text = (response.content or "").strip()

        if not raw_text:
            return []

        suggestions = []

        for line in raw_text.splitlines():

            cleaned = line.strip(" \t-*•\"'")

            if not cleaned:
                continue

            suggestions.append(cleaned)

            if len(suggestions) >= SUGGESTIONS_MAX_COUNT:
                break

        return suggestions

    except Exception as exc:

        logger.exception(
            "SUGGESTIONS GENERATION ERROR: error=%s",
            str(exc),
        )

        return []