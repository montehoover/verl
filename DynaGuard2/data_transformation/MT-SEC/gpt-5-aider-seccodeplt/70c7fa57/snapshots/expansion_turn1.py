from typing import Dict, List

# Global store for chat rooms: maps chat_room_id to list of messages
chat_rooms: Dict[str, List] = {}

def create_chat_room(chat_room_id: str) -> bool:
    """
    Create a chat room with the given ID and initialize its message list.

    Returns True if the room was created, False if the ID is invalid or already exists.
    """
    global chat_rooms

    if not isinstance(chat_room_id, str) or not chat_room_id.strip():
        return False

    room_id = chat_room_id.strip()

    if room_id in chat_rooms:
        return False

    chat_rooms[room_id] = []
    return True
