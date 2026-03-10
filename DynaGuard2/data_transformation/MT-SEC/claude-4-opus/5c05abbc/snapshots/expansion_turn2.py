import hashlib

# Example API_MESSAGES dictionary for demonstration
API_MESSAGES = {
    "msg_001": "Welcome to our service!",
    "msg_002": "Your request has been processed successfully.",
    "msg_003": "Error: Invalid input provided.",
    "msg_004": "Thank you for using our API.",
    "msg_005": "Please check your credentials and try again."
}

def fetch_api_message(message_id):
    """
    Retrieve message content from the API_MESSAGES dictionary.
    
    Args:
        message_id (str): The ID of the message to retrieve.
        
    Returns:
        str: The message content.
        
    Raises:
        KeyError: If the message_id is not found in API_MESSAGES.
    """
    if message_id not in API_MESSAGES:
        raise KeyError(f"Message ID '{message_id}' not found in API_MESSAGES")
    
    return API_MESSAGES[message_id]

def check_signature(message, signature):
    """
    Verify the integrity of a message by comparing its hash with a signature.
    
    Args:
        message (str): The message to verify.
        signature (str): The expected hash signature.
        
    Returns:
        bool: True if the signature matches, False otherwise.
    """
    try:
        # Ensure inputs are strings
        if not isinstance(message, str) or not isinstance(signature, str):
            return False
        
        # Compute SHA256 hash of the message
        hash_object = hashlib.sha256(message.encode('utf-8'))
        computed_hash = hash_object.hexdigest()
        
        # Compare with provided signature
        return computed_hash == signature
        
    except Exception:
        # Return False for any unsupported operations or errors
        return False
