from typing import Dict, List

chat_rooms_storage: Dict[str, List[str]] = {}


def create_chat_room(chat_room_id: str) -> bool:
    """
    Initializes an empty list for storing messages in the specified chat room.

    Args:
        chat_room_id: The ID of the chat room to create.

    Returns:
        True if the chat room was successfully created, False otherwise.
    """
    if chat_room_id not in chat_rooms_storage:
        chat_rooms_storage[chat_room_id] = []
        return True
    return False
