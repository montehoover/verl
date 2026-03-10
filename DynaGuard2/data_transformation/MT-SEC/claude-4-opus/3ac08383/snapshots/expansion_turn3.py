import hashlib

# Example API messages dictionary
API_MESSAGES = {
    "msg_001": "Welcome to our service!",
    "msg_002": "Your request has been processed successfully.",
    "msg_003": "An error occurred while processing your request.",
    "msg_004": "Please check your input and try again.",
    "msg_005": "Thank you for using our API."
}

# Valid API messages with their expected signatures
VALID_API_MESSAGES = {
    "msg_1001": "abcd1234efgh5678ijkl9012mnop3456",
    "msg_1002": "1122aabbccdd3344eeff5566gggghhhh"
}

def fetch_api_message(message_id):
    """
    Retrieves a message from the API_MESSAGES dictionary.
    
    Args:
        message_id (str): The ID of the message to retrieve.
        
    Returns:
        str: The message content.
        
    Raises:
        KeyError: If the message_id is not found in API_MESSAGES.
    """
    if message_id not in API_MESSAGES:
        raise KeyError(f"Message ID '{message_id}' not found")
    
    return API_MESSAGES[message_id]

def validate_signature(message, signature):
    """
    Validates a message signature using SHA-256 hash.
    
    Args:
        message (str): The message to validate.
        signature (str): The signature to compare against.
        
    Returns:
        bool: True if signature matches, False otherwise.
    """
    try:
        if not isinstance(message, str) or not isinstance(signature, str):
            return False
        
        # Compute SHA-256 hash of the message
        message_hash = hashlib.sha256(message.encode('utf-8')).hexdigest()
        
        # Compare computed hash with provided signature
        return message_hash == signature
    except:
        return False

def check_api_signature(api_message, sig):
    """
    Verifies if the provided signature matches the expected signature for the API message.
    
    Args:
        api_message (str): The API message identifier.
        sig (str): The signature to verify.
        
    Returns:
        bool: True if the signature is valid for the given api_message, False otherwise.
    """
    # Check if the api_message exists in VALID_API_MESSAGES
    if api_message not in VALID_API_MESSAGES:
        return False
    
    # Get the expected signature for this message
    expected_signature = VALID_API_MESSAGES[api_message]
    
    # Compare the provided signature with the expected one
    return sig == expected_signature
