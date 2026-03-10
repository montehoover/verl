chat_rooms_storage = {}

def send_message_to_chat(chat_room_id: str, sender: str, chat_message: str) -> bool:
    # Initialize chat room if it doesn't exist
    if chat_room_id not in chat_rooms_storage:
        chat_rooms_storage[chat_room_id] = []
    
    # Check if chat room has reached message limit (e.g., 1000 messages)
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
