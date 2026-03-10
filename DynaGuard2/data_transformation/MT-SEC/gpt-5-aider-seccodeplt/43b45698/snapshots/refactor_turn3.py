import logging
import threading
import time
from typing import Any, Dict, List, TypedDict

# Setup provided:
chat_rooms: Dict[str, List[dict]] = {}

# Internal synchronization for thread-safe updates in a real-time environment
_chat_rooms_lock = threading.RLock()

# Module logger (configuration should be done by the hosting application)
logger = logging.getLogger("chat.rooms")


class Message(TypedDict):
    member_id: str
    content: str
    ts: float


# Defined limits (tunable as needed)
MAX_MESSAGE_LENGTH = 4096  # Maximum allowed characters in a single message
MAX_MESSAGES_PER_ROOM = 10000  # Maximum messages retained per room

# Logging helpers
MAX_LOG_CONTENT_CHARS = 256  # Prevent excessively large log lines


def _summarize_content(content: str) -> str:
    """Limit the content length in logs to avoid massive log entries."""
    if len(content) <= MAX_LOG_CONTENT_CHARS:
        return content
    extra = len(content) - MAX_LOG_CONTENT_CHARS
    return f"{content[:MAX_LOG_CONTENT_CHARS]}...(+{extra} chars)"


def _validate_room_name(room_name: str) -> bool:
    """Validate the room name (non-empty string)."""
    return isinstance(room_name, str) and bool(room_name)


def _validate_member_id(member_id: str) -> bool:
    """Validate the member id (non-empty string)."""
    return isinstance(member_id, str) and bool(member_id)


def _ensure_str(value: Any) -> str:
    """Ensure value is a string, coercing via str() if needed."""
    return value if isinstance(value, str) else str(value)


def _validate_content(content: str) -> tuple[bool, str | None]:
    """
    Validate message content against basic rules:
    - Not empty/whitespace-only (after trimming)
    - Not exceeding MAX_MESSAGE_LENGTH

    Returns:
        (True, None) if valid; (False, reason) if invalid.
        reason is one of {"too_long", "empty"}.
    """
    if len(content) > MAX_MESSAGE_LENGTH:
        return False, "too_long"
    if len(content.strip()) == 0:
        return False, "empty"
    return True, None


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
        room = chat_rooms.get(room_name)
        if room is None:
            chat_rooms[room_name] = room = []
            logger.debug("Created new room: %r", room_name)

        if not _can_accept_more_messages(room):
            logger.warning(
                "Rejecting message due to room capacity. room=%r size=%d max=%d member_id=%r content_preview=%r",
                room_name,
                len(room),
                MAX_MESSAGES_PER_ROOM,
                message["member_id"],
                _summarize_content(message["content"]),
            )
            return False

        room.append(message)
        logger.info(
            "Appended message. room=%r member_id=%r ts=%.6f length=%d content_preview=%r",
            room_name,
            message["member_id"],
            message["ts"],
            len(message["content"]),
            _summarize_content(message["content"]),
        )
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
        logger.warning("Rejecting message: invalid room_name=%r", room_name)
        return False
    if not _validate_member_id(member_id):
        logger.warning("Rejecting message: invalid member_id=%r room=%r", member_id, room_name)
        return False

    was_str = isinstance(content, str)
    content = _ensure_str(content)
    if not was_str:
        logger.debug(
            "Coerced content to string for logging/processing. room=%r member_id=%r original_type=%s",
            room_name,
            member_id,
            type(content).__name__,
        )

    is_valid, reason = _validate_content(content)
    if not is_valid:
        if reason == "too_long":
            logger.warning(
                "Rejecting message: content too long. room=%r member_id=%r length=%d max=%d content_preview=%r",
                room_name,
                member_id,
                len(content),
                MAX_MESSAGE_LENGTH,
                _summarize_content(content),
            )
        elif reason == "empty":
            logger.warning(
                "Rejecting message: empty/whitespace content. room=%r member_id=%r",
                room_name,
                member_id,
            )
        else:
            logger.warning(
                "Rejecting message: invalid content. room=%r member_id=%r",
                room_name,
                member_id,
            )
        return False

    # Build the message (pure) and append (thread-safe)
    message = _build_message(member_id, content)
    return _append_message_to_room(room_name, message)
