# Dictionary to store message IDs
LOCAL_MESSAGES = {
    "msg_001": "Welcome message",
    "msg_002": "Error message",
    "msg_003": "Success message",
    "msg_004": "Warning message"
}

def check_message_existence(message_id):
    """
    Check if a given message ID exists in the local dictionary.
    
    Args:
        message_id (str): The message ID to check
        
    Returns:
        bool: True if the message ID exists, False otherwise
    """
    return message_id in LOCAL_MESSAGES
