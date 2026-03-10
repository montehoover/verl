chat_rooms = {}

def validate_message(text_message: str) -> bool:
    """Validate if a message meets the requirements."""
    return len(text_message) <= 1000

def create_message_entry(author_identifier: str, text_message: str) -> dict:
    """Create a message entry dictionary."""
    return {
        'author': author_identifier,
        'message': text_message
    }

def add_message_to_room(target_room: str, message_entry: dict) -> None:
    """Add a message entry to the specified chat room."""
    if target_room not in chat_rooms:
        chat_rooms[target_room] = []
    chat_rooms[target_room].append(message_entry)

def log_chat_message(target_room: str, author_identifier: str, text_message: str) -> bool:
    # Validate message
    if not validate_message(text_message):
        return False
    
    # Create message entry
    message_entry = create_message_entry(author_identifier, text_message)
    
    # Add message to room
    add_message_to_room(target_room, message_entry)
    
    return True
