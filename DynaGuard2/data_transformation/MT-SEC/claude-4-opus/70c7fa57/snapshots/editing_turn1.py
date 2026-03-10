# Global list to store chat messages
chat_messages = []

def log_message(author, content):
    """
    Logs a message to the chat room.
    
    Args:
        author (str): The author of the message
        content (str): The content of the message
    
    Returns:
        bool: True if message is logged successfully, False otherwise
    """
    try:
        message = {
            'author': author,
            'content': content
        }
        chat_messages.append(message)
        return True
    except:
        return False
