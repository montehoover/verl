from typing import Dict, List

# Global store for chat rooms: maps chat_room_id to list of messages
chat_rooms: Dict[str, List] = {}

# Maximum allowed length for a message
MAX_MESSAGE_LENGTH: int = 500

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

def validate_message(message_content: str) -> bool:
    """
    Validate that the message content is a string and does not exceed the maximum allowed length.

    Returns True if valid, False otherwise.
    """
    if not isinstance(message_content, str):
        return False

    if len(message_content) > MAX_MESSAGE_LENGTH:
        return False

    return True

def insert_chat_message(chat_room_id: str, author_id: str, message_content: str) -> bool:
    """
    Insert a message into the specified chat room's conversation log.

    Returns True on success, False if validation fails or the chat room does not exist.
    """
    global chat_rooms

    # Validate inputs
    if not isinstance(chat_room_id, str) or not chat_room_id.strip():
        return False
    if not isinstance(author_id, str) or not author_id.strip():
        return False
    if not validate_message(message_content):
        return False

    room_id = chat_room_id.strip()

    # Ensure the chat room exists and is a list for messages
    if room_id not in chat_rooms or not isinstance(chat_rooms[room_id], list):
        return False

    # Construct the message record
    message = {
        "author_id": author_id.strip(),
        "content": message_content
    }

    # Append the message to the chat room history
    chat_rooms[room_id].append(message)
    return True
