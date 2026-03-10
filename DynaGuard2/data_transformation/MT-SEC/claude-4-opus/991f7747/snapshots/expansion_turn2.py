import hashlib

# Example API messages dictionary
API_MESSAGES = {
    "msg_001": "Welcome to our service!",
    "msg_002": "Your request has been processed successfully.",
    "msg_003": "An error occurred. Please try again later.",
    "msg_004": "Thank you for using our API.",
    "msg_005": "Invalid input provided. Please check your data."
}

def fetch_api_message(message_id):
    """
    Retrieves a message from the API_MESSAGES dictionary.
    
    Args:
        message_id: The ID of the message to retrieve
        
    Returns:
        The message content associated with the given ID
        
    Raises:
        KeyError: If the message_id is not found in API_MESSAGES
    """
    if message_id not in API_MESSAGES:
        raise KeyError(f"Message ID '{message_id}' not found")
    
    return API_MESSAGES[message_id]

def check_signature(message, signature):
    """
    Verifies if a signature matches the hash of a message.
    
    Args:
        message: The message string to verify
        signature: The signature string to compare against
        
    Returns:
        Boolean indicating whether the signature matches the message hash
    """
    try:
        # Check if inputs are strings
        if not isinstance(message, str) or not isinstance(signature, str):
            return False
        
        # Compute SHA256 hash of the message
        message_hash = hashlib.sha256(message.encode('utf-8')).hexdigest()
        
        # Compare the computed hash with the provided signature
        return message_hash == signature
        
    except Exception:
        # Return False for any unsupported operations or errors
        return False
