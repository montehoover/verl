from typing import Dict, List, Union

chat_rooms_storage: Dict[str, List[Union[str, Dict[str, str]]]] = {}


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


def send_message_to_chat(chat_room_id: str, sender: str, chat_message: str) -> bool:
    """
    Appends a new message as a dictionary to a specific chat room's history.

    Args:
        chat_room_id: The ID of the chat room.
        sender: The sender of the message.
        chat_message: The content of the message.

    Returns:
        True if the message was successfully sent, False otherwise (e.g., room doesn't exist or message limit reached).
    """
    if chat_room_id not in chat_rooms_storage:
        return False
    if len(chat_rooms_storage[chat_room_id]) >= 100:  # Assuming a limit of 100 messages per room
        return False
    chat_rooms_storage[chat_room_id].append({
        "sender": sender,
        "message": chat_message
    })
    return True
