# Global list to store messages
message_log = []

def log_message(message):
    """
    Logs a message to the global message list.
    
    Args:
        message (str): The message to log
        
    Returns:
        bool: True if message was logged successfully, False otherwise
    """
    try:
        message_log.append(message)
        return True
    except:
        return False
