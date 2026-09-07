import time
from threading import Lock


# Temporary in-memory conversation location storage.
# Key: conversation_id
# Value: location data + expiry timestamp
_location_store: dict[str, dict] = {}

_store_lock = Lock()

# Location remains available for 1 hour after the last update/access.
LOCATION_TTL_SECONDS = 60 * 60


def set_conversation_location(
    conversation_id: str,
    latitude: float,
    longitude: float,
    address: str | None = None,
) -> None:
    """
    Temporarily store location for a conversation.
    This is NOT persisted in the database.
    """
    now = time.time()

    with _store_lock:
        _location_store[str(conversation_id)] = {
            "latitude": latitude,
            "longitude": longitude,
            "address": address,
            "expires_at": now + LOCATION_TTL_SECONDS,
        }


def get_conversation_location(
    conversation_id: str,
) -> dict | None:
    """
    Return the temporarily stored location for a conversation.

    Returns None if:
    - no location exists
    - location has expired
    """
    conversation_id = str(conversation_id)
    now = time.time()

    with _store_lock:
        location = _location_store.get(conversation_id)

        if not location:
            return None

        if location["expires_at"] <= now:
            del _location_store[conversation_id]
            return None

        return {
            "latitude": location["latitude"],
            "longitude": location["longitude"],
            "address": location.get("address"),
        }


def delete_conversation_location(
    conversation_id: str,
) -> None:
    """
    Remove temporary location for a conversation.
    """
    with _store_lock:
        _location_store.pop(str(conversation_id), None)