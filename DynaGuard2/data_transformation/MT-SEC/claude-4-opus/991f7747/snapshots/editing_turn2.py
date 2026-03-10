# Dictionary to store message IDs
LOCAL_MESSAGES = {
    "msg_001": "Welcome message",
    "msg_002": "Error message",
    "msg_003": "Success message",
    "msg_004": "Warning message"
}

def check_message_existence(message_id, expected_value):
    """
    Check if a given message ID exists in the local dictionary and matches the expected value.
    
    Args:
        message_id (str): The message ID to check
        expected_value (str): The expected value to compare against
        
    Returns:
        bool: True if the message ID exists and matches the expected value, False otherwise
    """
    return message_id in LOCAL_MESSAGES and LOCAL_MESSAGES[message_id] == expected_value
