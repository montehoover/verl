# Global list to store messages
message_log = []

def log_message(message, sender_id, timestamp):
    """
    Logs a message to the global message list.
    
    Args:
        message (str): The message to log
        sender_id (str): The ID of the sender
        timestamp (str): The timestamp of the message
        
    Returns:
        bool: True if message was logged successfully, False otherwise
    """
    try:
        message_entry = {
            'message': message,
            'sender_id': sender_id,
            'timestamp': timestamp
        }
        message_log.append(message_entry)
        return True
    except:
        return False
