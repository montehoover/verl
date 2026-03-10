import threading
import datetime
from typing import Dict, List, TypedDict, Optional

# Global chat room repository
# Maps room identifier -> list of message entries
chat_rooms: Dict[str, List["MessageEntry"]] = {}

# Optional concurrency guard for real-time updates
_chat_rooms_lock = threading.RLock()

# Established limits (can be tuned as needed)
MAX_MESSAGE_LENGTH = 10_000  # Maximum number of characters allowed per message
MAX_MESSAGES_PER_ROOM = 10_000  # Maximum number of messages stored per room


class MessageEntry(TypedDict):
    author: str
    text: str
    timestamp: str  # ISO 8601 in UTC


def normalize_to_str(value) -> str:
    """
    Pure helper to ensure the value is a string.
    """
    return value if isinstance(value, str) else str(value)


def validate_message_content(
    text_message: str,
    max_message_length: Optional[int] = MAX_MESSAGE_LENGTH,
) -> bool:
    """
    Pure validation for message content constraints (e.g., length).

    Returns True if message content is valid, False otherwise.
    """
    if max_message_length is not None and len(text_message) > max_message_length:
        return False
    return True


def get_current_utc_iso_timestamp() -> str:
    """
    Returns the current UTC timestamp in ISO 8601 format.
    """
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def build_message_entry(author_identifier: str, text_message: str, timestamp: str) -> MessageEntry:
    """
    Pure builder for a MessageEntry dictionary.
    """
    return MessageEntry(author=author_identifier, text=text_message, timestamp=timestamp)


def try_log_message_to_room(
    target_room: str,
    entry: MessageEntry,
    max_messages_per_room: Optional[int] = MAX_MESSAGES_PER_ROOM,
) -> bool:
    """
    Impure helper that appends a message entry to the room atomically with respect to limits.

    Returns True if appended, False if the room is at capacity.
    """
    with _chat_rooms_lock:
        room_history = chat_rooms.setdefault(target_room, [])
        if max_messages_per_room is not None and len(room_history) >= max_messages_per_room:
            return False
        room_history.append(entry)
        return True


def log_chat_message(target_room: str, author_identifier: str, text_message: str) -> bool:
    """
    Integrates a new message into the chat room's conversation log.

    Args:
        target_room: A distinctive string code identifying the target chat room.
        author_identifier: A unique string identifier for the message author.
        text_message: The textual content of the message to be integrated.

    Returns:
        True if the message was successfully incorporated into the chat room,
        False if the message was rejected for exceeding established limits.
    """
    # Normalize inputs (pure)
    target_room = normalize_to_str(target_room)
    author_identifier = normalize_to_str(author_identifier)
    text_message = normalize_to_str(text_message)

    # Validate message content (pure)
    if not validate_message_content(text_message, MAX_MESSAGE_LENGTH):
        return False

    # Build message entry (pure builder + time fetch)
    timestamp = get_current_utc_iso_timestamp()
    entry = build_message_entry(author_identifier, text_message, timestamp)

    # Log to room with capacity check (impure)
    return try_log_message_to_room(target_room, entry, MAX_MESSAGES_PER_ROOM)
