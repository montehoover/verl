from typing import Dict, List

# Global dictionary mapping room IDs to their message lists
chat_rooms: Dict[str, List[Dict[str, str]]] = {}

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

def add_message_to_room(room_id: str, author: str, message: str) -> bool:
    """
    Add a message to the specified chat room.

    Returns True if the message was added successfully, False otherwise.
    """
    # Validate types
    if not all(isinstance(x, str) for x in (room_id, author, message)):
        return False

    # Normalize and validate contents
    room_id = room_id.strip()
    author = author.strip()
    message = message.strip()

    if not room_id or not author or not message:
        return False

    # Ensure the room exists
    if room_id not in chat_rooms:
        return False

    # Append structured message
    chat_rooms[room_id].append({"author": author, "message": message})
    return True

def log_chat_message(target_room: str, author_identifier: str, text_message: str) -> bool:
    """
    Integrate a message into a chat room's conversation log.

    Returns True if the message was added successfully, False otherwise.
    """
    # Validate types
    if not all(isinstance(x, str) for x in (target_room, author_identifier, text_message)):
        return False

    # Normalize inputs
    target_room = target_room.strip()
    author_identifier = author_identifier.strip()
    text_message = text_message.strip()

    # Validate non-empty fields
    if not target_room or not author_identifier or not text_message:
        return False

    # Delegate to existing message addition logic
    return add_message_to_room(target_room, author_identifier, text_message)
