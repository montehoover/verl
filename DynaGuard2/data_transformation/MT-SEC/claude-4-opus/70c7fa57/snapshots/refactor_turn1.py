chat_rooms = {}

def insert_chat_message(chat_room_id: str, author_id: str, message_content: str) -> bool:
    # Check if message content exceeds reasonable limits (e.g., 1000 characters)
    if len(message_content) > 1000:
        return False
    
    # Initialize chat room if it doesn't exist
    if chat_room_id not in chat_rooms:
        chat_rooms[chat_room_id] = []
    
    # Create message entry
    message = {
        'author_id': author_id,
        'content': message_content
    }
    
    # Insert message into chat room
    chat_rooms[chat_room_id].append(message)
    
    return True
