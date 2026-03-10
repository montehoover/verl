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
