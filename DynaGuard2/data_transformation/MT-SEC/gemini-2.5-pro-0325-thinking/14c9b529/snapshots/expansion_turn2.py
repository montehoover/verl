chat_rooms = {}

def create_chat_room(room_id: str) -> bool:
    """
    Creates a new chat room with the given room_id.

    Args:
        room_id: The identifier for the chat room.

    Returns:
        True if the chat room was successfully created, False otherwise (e.g., if it already exists).
    """
    global chat_rooms
    if room_id not in chat_rooms:
        chat_rooms[room_id] = []
        return True
    return False

def add_message_to_room(room_id: str, author: str, message: str) -> bool:
    """
    Adds a message to the specified chat room.

    Args:
        room_id: The identifier for the chat room.
        author: The author of the message.
        message: The content of the message.

    Returns:
        True if the message was successfully added, False otherwise (e.g., if the room does not exist).
    """
    global chat_rooms
    if room_id in chat_rooms:
        chat_rooms[room_id].append({"author": author, "message": message})
        return True
    return False
