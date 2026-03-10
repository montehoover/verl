# Local dictionary to store messages
messages = {}

def is_message_valid(message_id, expected_signature):
    """
    Check if a given message identifier exists in the local dictionary
    and verify its signature matches the expected signature.
    
    Args:
        message_id (str): The message identifier to check
        expected_signature (str): The expected signature to verify
        
    Returns:
        bool: True if the message exists and signature matches, False otherwise
    """
    if message_id in messages:
        return messages[message_id] == expected_signature
    return False
