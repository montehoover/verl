import hashlib

# Example API messages dictionary
API_MESSAGES = {
    "msg_001": "Welcome to our service!",
    "msg_002": "Your request has been processed successfully.",
    "msg_003": "An error occurred while processing your request.",
    "msg_004": "Please check your input and try again.",
    "msg_005": "Thank you for using our API."
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
