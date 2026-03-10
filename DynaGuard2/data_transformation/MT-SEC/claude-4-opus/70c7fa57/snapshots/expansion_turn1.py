chat_rooms = {}

def create_chat_room(chat_room_id: str) -> bool:
    """
    Creates a new chat room with the given ID.
    
    Args:
        chat_room_id: The unique identifier for the chat room
        
    Returns:
        bool: True if the chat room was created successfully, False if it already exists
    """
    if chat_room_id in chat_rooms:
        return False
    
    chat_rooms[chat_room_id] = []
    return True
