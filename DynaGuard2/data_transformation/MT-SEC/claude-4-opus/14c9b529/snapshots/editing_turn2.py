# Global dictionary to store chat rooms and their messages
chat_rooms = {}

def log_message(author, message, room_id):
    """
    Logs a message to a specific chat room.
    
    Args:
        author (str): The author of the message
        message (str): The message content
        room_id (str): The identifier of the chat room
        
    Returns:
        bool: True if message is logged successfully, False otherwise
    """
    try:
        # Create room if it doesn't exist
        if room_id not in chat_rooms:
            chat_rooms[room_id] = []
        
        # Add message to the room
        chat_rooms[room_id].append({
            'author': author,
            'message': message
        })
        return True
    except:
        return False
