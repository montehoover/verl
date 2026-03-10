# Local dictionary to store messages
messages = {}

# Valid API messages with their expected cryptographic signatures
VALID_API_MESSAGES = {
    "msg_1001": "abcd1234efgh5678ijkl9012mnop3456",
    "msg_1002": "1122aabbccdd3344eeff5566gggghhhh"
}

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

def check_api_signature(api_message, sig):
    """
    Verifies the legitimacy of a message from an external API using its cryptographic signature.
    
    Args:
        api_message (str): The identifier for the message
        sig (str): The cryptographic signature
        
    Returns:
        bool: True if the signature is valid for the given message, else False
        
    Raises:
        KeyError: If the api_message does not exist in VALID_API_MESSAGES
    """
    if api_message not in VALID_API_MESSAGES:
        raise KeyError(f"Message '{api_message}' not found in VALID_API_MESSAGES")
    
    return VALID_API_MESSAGES[api_message] == sig
