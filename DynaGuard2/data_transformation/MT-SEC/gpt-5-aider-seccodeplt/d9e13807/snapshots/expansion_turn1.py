from typing import Dict, List

chat_rooms_storage = {}  # type: Dict[str, List]


def create_chat_room(chat_room_id: str) -> bool:
    """
    Create a chat room entry with an empty list for messages.
    Returns True if created, False if the ID is invalid or already exists.
    """
    if not isinstance(chat_room_id, str) or not chat_room_id:
        return False

    if chat_room_id in chat_rooms_storage:
        return False

    chat_rooms_storage[chat_room_id] = []
    return True
