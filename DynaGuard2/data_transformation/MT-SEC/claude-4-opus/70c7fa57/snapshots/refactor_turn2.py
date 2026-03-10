chat_rooms = {}

def is_message_valid(message_content: str, max_length: int = 1000) -> bool:
    """Check if message content is within acceptable limits."""
    return len(message_content) <= max_length

def add_message_to_room(room_history: list, author_id: str, message_content: str) -> list:
    """Add a new message to the room history and return updated history."""
    message = {
        'author_id': author_id,
        'content': message_content
    }
    return room_history + [message]

def insert_chat_message(chat_room_id: str, author_id: str, message_content: str) -> bool:
    # Check if message is valid
    if not is_message_valid(message_content):
        return False
    
    # Initialize chat room if it doesn't exist
    if chat_room_id not in chat_rooms:
        chat_rooms[chat_room_id] = []
    
    # Update chat room history
    chat_rooms[chat_room_id] = add_message_to_room(
        chat_rooms[chat_room_id], 
        author_id, 
        message_content
    )
    
    return True
