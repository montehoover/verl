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

def validate_message(message_content: str) -> bool:
    """
    Validates if a message's length is within acceptable limits.
    
    Args:
        message_content: The message content to validate
        
    Returns:
        bool: True if the message length is valid (<=500 chars), False otherwise
    """
    return len(message_content) <= 500

def insert_chat_message(chat_room_id: str, author_id: str, message_content: str) -> bool:
    """
    Inserts a message into the specified chat room's history.
    
    Args:
        chat_room_id: The identifier of the chat room
        author_id: The identifier of the message author
        message_content: The content of the message
        
    Returns:
        bool: True if the message was inserted successfully, False otherwise
    """
    if chat_room_id not in chat_rooms:
        return False
    
    if not validate_message(message_content):
        return False
    
    message = {
        'author_id': author_id,
        'content': message_content
    }
    
    chat_rooms[chat_room_id].append(message)
    return True
