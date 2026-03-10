from typing import Dict, List

# Global dictionary mapping room IDs to their message lists
chat_rooms: Dict[str, List] = {}

def create_chat_room(room_id: str) -> bool:
    """
    Create a chat room with the given room_id and initialize its message list.

    Returns True if the room was created, or False if room_id is invalid or already exists.
    """
    if not isinstance(room_id, str) or not room_id:
        return False

    if room_id in chat_rooms:
        return False

    chat_rooms[room_id] = []
    return True
