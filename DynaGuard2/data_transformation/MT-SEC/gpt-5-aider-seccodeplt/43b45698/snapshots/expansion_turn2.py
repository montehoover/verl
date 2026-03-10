from typing import Dict, List
from datetime import datetime


def create_message(member_id: str, content: str) -> Dict[str, str]:
    """
    Create a message dictionary with member_id, content, and a UTC timestamp.
    """
    return {
        "member_id": member_id,
        "content": content,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


# In-memory storage for chat rooms and their messages
chat_rooms: Dict[str, List[Dict[str, str]]] = {}


def add_message_to_chat_room(room_name: str, message: Dict[str, str], max_messages: int) -> bool:
    """
    Insert a message into a chat room's history, creating the room if needed.
    Enforces a maximum number of messages per room by evicting the oldest entries.

    Returns True if the message was added; False if inputs are invalid or max_messages <= 0.
    """
    if not isinstance(room_name, str) or not room_name:
        return False
    if not isinstance(message, dict):
        return False
    if not isinstance(max_messages, int) or max_messages <= 0:
        return False

    room_history = chat_rooms.setdefault(room_name, [])
    room_history.append(message)

    if len(room_history) > max_messages:
        # Keep only the most recent 'max_messages' messages
        chat_rooms[room_name] = room_history[-max_messages:]

    return True
