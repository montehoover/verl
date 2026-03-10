# Local dictionary to store messages
messages = {}

def check_message_existence(message_id, valid_ids):
    """Check if a given message ID exists in the list of valid IDs.
    
    Args:
        message_id (str): The message ID to check
        valid_ids (list): List of valid message IDs
        
    Returns:
        bool: True if the message ID is found in valid_ids, otherwise False
    """
    return message_id in valid_ids
