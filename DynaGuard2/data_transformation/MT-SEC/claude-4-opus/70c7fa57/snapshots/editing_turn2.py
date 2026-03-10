# Global dictionary to store chat messages for multiple rooms
chat_rooms = {}

def log_message(room_id, author, content):
    """
    Logs a message to a specific chat room.
    
    Args:
        room_id (str): The ID of the chat room
        author (str): The author of the message
        content (str): The content of the message
    
    Returns:
        bool: True if message is logged successfully, False otherwise
    """
    try:
        # Create room if it doesn't exist
        if room_id not in chat_rooms:
            chat_rooms[room_id] = []
        
        message = {
            'author': author,
            'content': content
        }
        chat_rooms[room_id].append(message)
        return True
    except:
        return False
