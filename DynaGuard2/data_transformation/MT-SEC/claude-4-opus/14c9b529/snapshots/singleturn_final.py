chat_rooms = {}

def log_chat_message(target_room: str, author_identifier: str, text_message: str) -> bool:
    """
    Integrates new messages into a chat room's conversation log.
    
    Args:
        target_room: A distinctive string code identifying the target chat room.
        author_identifier: A unique string identifier for the message author.
        text_message: The textual content of the message to be integrated.
    
    Returns:
        bool: True if the message was successfully incorporated into the chat room,
              False if the message was rejected for exceeding established limits.
    """
    # Define message length limit
    MAX_MESSAGE_LENGTH = 1000
    
    # Check if message exceeds length limit
    if len(text_message) > MAX_MESSAGE_LENGTH:
        return False
    
    # Initialize chat room if it doesn't exist
    if target_room not in chat_rooms:
        chat_rooms[target_room] = []
    
    # Create message entry
    message_entry = {
        'author': author_identifier,
        'message': text_message
    }
    
    # Add message to the chat room
    chat_rooms[target_room].append(message_entry)
    
    return True
