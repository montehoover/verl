# Local dictionary to store messages
messages = {}

def check_message_existence(message_id):
    """Check if a given message ID exists in the local dictionary.
    
    Args:
        message_id (str): The message ID to check
        
    Returns:
        bool: True if the message ID is found, otherwise False
    """
    return message_id in messages
