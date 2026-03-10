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


def add_message_to_room(chat_room_id: str, sender: str, message: str) -> bool:
    """
    Adds a message to the specified chat room's message list.

    Args:
        chat_room_id: The ID of the chat room.
        sender: The sender of the message.
        message: The message content.

    Returns:
        True if the message was successfully added, False otherwise.
    """
    if chat_room_id in chat_rooms_storage:
        formatted_message = f"{sender}: {message}"
        chat_rooms_storage[chat_room_id].append(formatted_message)
        return True
    return False
