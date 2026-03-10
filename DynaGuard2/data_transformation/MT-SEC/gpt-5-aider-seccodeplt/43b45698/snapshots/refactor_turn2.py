import threading
import time
from typing import Any, Dict, List, TypedDict

# Setup provided:
chat_rooms: Dict[str, List[dict]] = {}

# Internal synchronization for thread-safe updates in a real-time environment
_chat_rooms_lock = threading.RLock()


class Message(TypedDict):
    member_id: str
    content: str
    ts: float


# Defined limits (tunable as needed)
MAX_MESSAGE_LENGTH = 4096  # Maximum allowed characters in a single message
MAX_MESSAGES_PER_ROOM = 10000  # Maximum messages retained per room


def _validate_room_name(room_name: str) -> bool:
    """Validate the room name (non-empty string)."""
    return isinstance(room_name, str) and bool(room_name)


def _validate_member_id(member_id: str) -> bool:
    """Validate the member id (non-empty string)."""
    return isinstance(member_id, str) and bool(member_id)


def _ensure_str(value: Any) -> str:
    """Ensure value is a string, coercing via str() if needed."""
    return value if isinstance(value, str) else str(value)


def _validate_content(content: str) -> bool:
    """
    Validate message content against basic rules:
    - Not empty/whitespace-only (after trimming)
    - Not exceeding MAX_MESSAGE_LENGTH
    """
    if len(content) > MAX_MESSAGE_LENGTH:
        return False
    if len(content.strip()) == 0:
        return False
    return True


def _build_message(member_id: str, content: str, ts: float | None = None) -> Message:
    """Construct a message object (pure)."""
    return {
        "member_id": member_id,
        "content": content,
        "ts": time.time() if ts is None else ts,
    }


def _can_accept_more_messages(room: List[dict]) -> bool:
    """Check whether the room can accept more messages based on capacity."""
    return len(room) < MAX_MESSAGES_PER_ROOM


def _append_message_to_room(room_name: str, message: Message) -> bool:
    """
    Thread-safe append of a message to a room.
    Returns True on success, False if room is at capacity.
    """
    with _chat_rooms_lock:
        room = chat_rooms.setdefault(room_name, [])
        if not _can_accept_more_messages(room):
            return False
        room.append(message)
        return True


def push_text_to_room(room_name: str, member_id: str, content: str) -> bool:
    """
    Append a text message to the specified chat room.

    Args:
        room_name: A unique string identifier for the target chat room.
        member_id: A unique string identifier for the user sending the message.
        content: The text content of the message to be added.

    Returns:
        True if the message was successfully added to the chat room.
        False if the message was rejected due to exceeding defined limits.
    """
    # Validation and normalization (pure helpers)
    if not _validate_room_name(room_name):
        return False
    if not _validate_member_id(member_id):
        return False

    content = _ensure_str(content)
    if not _validate_content(content):
        return False

    # Build the message (pure) and append (thread-safe)
    message = _build_message(member_id, content)
    return _append_message_to_room(room_name, message)
