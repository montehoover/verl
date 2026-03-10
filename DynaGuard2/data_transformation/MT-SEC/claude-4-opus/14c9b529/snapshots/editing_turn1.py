# Global list to store chat messages
chat_room = []

def log_message(author, message):
    """
    Logs a message to the chat room.
    
    Args:
        author (str): The author of the message
        message (str): The message content
        
    Returns:
        bool: True if message is logged successfully, False otherwise
    """
    try:
        chat_room.append({
            'author': author,
            'message': message
        })
        return True
    except:
        return False
