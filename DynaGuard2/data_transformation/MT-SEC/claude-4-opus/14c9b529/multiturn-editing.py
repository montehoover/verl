# Global dictionary to store chat rooms and their messages
chat_rooms = {}

# Configuration for message limits
MAX_MESSAGES_PER_ROOM = 1000
MAX_MESSAGE_LENGTH = 5000

def log_chat_message(target_room, author_identifier, text_message):
    """
    Logs a message to a specific chat room with message limit handling.
    
    Args:
        target_room (str): The chat room ID
        author_identifier (str): The message author
        text_message (str): The message content
        
    Returns:
        bool: True if message is logged successfully, False otherwise
    """
    try:
        # Validate message length
        if len(text_message) > MAX_MESSAGE_LENGTH:
            return False
            
        # Create room if it doesn't exist
        if target_room not in chat_rooms:
            chat_rooms[target_room] = []
        
        # Check message limit for the room
        if len(chat_rooms[target_room]) >= MAX_MESSAGES_PER_ROOM:
            # Remove oldest message to make room
            chat_rooms[target_room].pop(0)
        
        # Add message to the room
        chat_rooms[target_room].append({
            'author': author_identifier,
            'message': text_message
        })
        return True
    except:
        return False
