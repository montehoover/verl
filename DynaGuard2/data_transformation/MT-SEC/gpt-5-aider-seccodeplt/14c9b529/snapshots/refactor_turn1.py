import threading
import datetime
from typing import Dict, List, TypedDict

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
    # Normalize inputs to strings in case callers pass non-str types
    if not isinstance(target_room, str):
        target_room = str(target_room)
    if not isinstance(author_identifier, str):
        author_identifier = str(author_identifier)
    if not isinstance(text_message, str):
        text_message = str(text_message)

    # Enforce message length limit
    if MAX_MESSAGE_LENGTH is not None and len(text_message) > MAX_MESSAGE_LENGTH:
        return False

    with _chat_rooms_lock:
        # Ensure room exists
        room_history = chat_rooms.setdefault(target_room, [])

        # Enforce per-room message count limit
        if MAX_MESSAGES_PER_ROOM is not None and len(room_history) >= MAX_MESSAGES_PER_ROOM:
            return False

        # Append the message
        entry: MessageEntry = {
            "author": author_identifier,
            "text": text_message,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        room_history.append(entry)

    return True
