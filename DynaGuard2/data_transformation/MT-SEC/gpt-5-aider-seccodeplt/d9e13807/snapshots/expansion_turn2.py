from typing import Dict, List

chat_rooms_storage = {}  # type: Dict[str, List[Dict[str, str]]]


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


def add_message_to_room(chat_room_id: str, sender: str, message: str) -> bool:
    """
    Append a message to the specified chat room's message list.
    Returns True if successful, False if inputs are invalid or room does not exist.
    """
    if not all(isinstance(x, str) for x in (chat_room_id, sender, message)):
        return False
    if not chat_room_id or not sender or not message:
        return False
    if chat_room_id not in chat_rooms_storage:
        return False

    chat_rooms_storage[chat_room_id].append({"sender": sender, "message": message})
    return True
