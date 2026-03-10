chat_rooms = {}

def insert_chat_message(chat_room_id: str, author_id: str, message_content: str) -> bool:
    """
    Integrates new messages into a chat room's conversation log.
    
    Args:
        chat_room_id: A distinctive string code identifying the target chat room.
        author_id: A unique string identifier for the message author.
        message_content: The textual content of the message to be integrated.
    
    Returns:
        bool: True if the message was successfully incorporated into the chat room,
              False if the message was rejected for exceeding established limits.
    """
    # Define message length limit
    MAX_MESSAGE_LENGTH = 1000
    
    # Check if message exceeds length limit
    if len(message_content) > MAX_MESSAGE_LENGTH:
        return False
    
    # Create chat room if it doesn't exist
    if chat_room_id not in chat_rooms:
        chat_rooms[chat_room_id] = []
    
    # Create message object
    message = {
        'author_id': author_id,
        'content': message_content
    }
    
    # Add message to chat room
    chat_rooms[chat_room_id].append(message)
    
    return True
