import threading
import time
from typing import Dict, List, TypedDict

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
    # Basic input validation and normalization
    if not isinstance(room_name, str) or not room_name:
        return False
    if not isinstance(member_id, str) or not member_id:
        return False
    if not isinstance(content, str):
        content = str(content)

    # Enforce message content limits
    trimmed = content.strip()
    if len(trimmed) == 0:
        # Reject empty/whitespace-only messages
        return False
    if len(content) > MAX_MESSAGE_LENGTH:
        return False

    # Thread-safe room creation and limit enforcement
    with _chat_rooms_lock:
        room = chat_rooms.setdefault(room_name, [])
        if len(room) >= MAX_MESSAGES_PER_ROOM:
            return False

        message: Message = {
            "member_id": member_id,
            "content": content,
            "ts": time.time(),
        }
        room.append(message)

    return True
