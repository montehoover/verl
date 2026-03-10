chat_rooms = {}

def log_chat_message(target_room: str, author_identifier: str, text_message: str) -> bool:
    # Check if message exceeds length limit (assuming 1000 characters as a reasonable limit)
    if len(text_message) > 1000:
        return False
    
    # Initialize chat room if it doesn't exist
    if target_room not in chat_rooms:
        chat_rooms[target_room] = []
    
    # Create message entry
    message_entry = {
        'author': author_identifier,
        'message': text_message
    }
    
    # Add message to chat room
    chat_rooms[target_room].append(message_entry)
    
    return True
