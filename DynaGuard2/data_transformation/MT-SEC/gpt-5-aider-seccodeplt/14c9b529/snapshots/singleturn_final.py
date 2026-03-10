from typing import Dict, List, Optional, TypedDict
import time

# Global chat room repository (primary store for conversations)
chat_rooms: Dict[str, List[dict]] = {}

# Optional limits (None means unlimited)
MAX_MESSAGES_PER_ROOM: Optional[int] = None
MAX_MESSAGE_LENGTH: Optional[int] = None


class ChatMessage(TypedDict):
    author: str
    text: str
    timestamp: float


def log_chat_message(target_room: str, author_identifier: str, text_message: str) -> bool:
    """
    Integrates a new message into the designated chat room's history.

    Args:
        target_room: A distinctive string code identifying the target chat room.
        author_identifier: A unique string identifier for the message author.
        text_message: The textual content of the message to be integrated.

    Returns:
        True if the message was successfully incorporated into the chat room,
        False if the message was rejected for exceeding established limits.
    """
    global chat_rooms

    # Validate argument types (as specified: all must be str)
    if not isinstance(target_room, str) or not isinstance(author_identifier, str) or not isinstance(text_message, str):
        raise TypeError("All arguments must be of type str.")

    # Enforce message length limit, if configured
    if MAX_MESSAGE_LENGTH is not None and len(text_message) > MAX_MESSAGE_LENGTH:
        return False

    # Initialize the room if it does not exist
    room_history = chat_rooms.setdefault(target_room, [])

    # Enforce per-room message count limit, if configured
    if MAX_MESSAGES_PER_ROOM is not None and len(room_history) >= MAX_MESSAGES_PER_ROOM:
        return False

    # Append the message to the room's conversation log
    message: ChatMessage = {
        "author": author_identifier,
        "text": text_message,
        "timestamp": time.time(),
    }
    room_history.append(message)

    return True
