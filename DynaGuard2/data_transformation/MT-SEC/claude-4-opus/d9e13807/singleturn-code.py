chat_rooms_storage = {}

def send_message_to_chat(chat_room_id: str, sender: str, chat_message: str) -> bool:
    """
    Appends a new message to a specific chat room's history.
    
    Args:
        chat_room_id: Unique identifier for the target chat room.
        sender: Unique identifier for the message sender.
        chat_message: The actual text content of the message.
    
    Returns:
        Returns True if the message was successfully added to the chat room,
        False if it was rejected due to exceeding limits.
    """
    # Initialize chat room if it doesn't exist
    if chat_room_id not in chat_rooms_storage:
        chat_rooms_storage[chat_room_id] = []
    
    # Check if message limit is exceeded (e.g., 1000 messages per room)
    if len(chat_rooms_storage[chat_room_id]) >= 1000:
        return False
    
    # Create message object
    message = {
        'sender': sender,
        'message': chat_message,
        'timestamp': None  # Could add timestamp if needed
    }
    
    # Append message to chat room
    chat_rooms_storage[chat_room_id].append(message)
    
    return True
