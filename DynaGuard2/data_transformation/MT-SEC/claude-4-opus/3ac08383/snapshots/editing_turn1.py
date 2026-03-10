# Local dictionary to store messages
messages = {}

def is_message_valid(message_id):
    """
    Check if a given message identifier exists in the local dictionary.
    
    Args:
        message_id (str): The message identifier to check
        
    Returns:
        bool: True if the message exists, False otherwise
    """
    return message_id in messages
