chat_rooms_storage = {}

def _validate_message_limit(chat_room_messages: list, limit: int = 1000) -> bool:
    """Check if the chat room has reached its message limit."""
    return len(chat_room_messages) < limit

def _create_message(sender: str, chat_message: str) -> dict:
    """Create a message object with sender and content."""
    return {
        'sender': sender,
        'message': chat_message,
        'timestamp': None  # Could add timestamp if needed
    }

def _get_or_create_chat_room(chat_room_id: str) -> list:
    """Get existing chat room or create a new one if it doesn't exist."""
    if chat_room_id not in chat_rooms_storage:
        chat_rooms_storage[chat_room_id] = []
    return chat_rooms_storage[chat_room_id]

def _add_message_to_room(chat_room_messages: list, message: dict) -> None:
    """Add a message to the chat room's message list."""
    chat_room_messages.append(message)

def send_message_to_chat(chat_room_id: str, sender: str, chat_message: str) -> bool:
    # Get or create chat room
    chat_room_messages = _get_or_create_chat_room(chat_room_id)
    
    # Validate message limit
    if not _validate_message_limit(chat_room_messages):
        return False
    
    # Create and add message
    message = _create_message(sender, chat_message)
    _add_message_to_room(chat_room_messages, message)
    
    return True
