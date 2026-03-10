import hashlib

# Example API_MESSAGES dictionary for demonstration
API_MESSAGES = {
    "msg_001": "Welcome to our service!",
    "msg_002": "Your order has been confirmed.",
    "msg_003": "Payment received successfully.",
    "msg_004": "Item shipped. Tracking number: XYZ123",
    "msg_005": "Thank you for your feedback!"
}

def fetch_message(message_id):
    """
    Retrieves message content from the API_MESSAGES dictionary.
    
    Args:
        message_id (str): The ID of the message to retrieve
        
    Returns:
        str: The message content
        
    Raises:
        KeyError: If the message_id is not found in API_MESSAGES
    """
    if message_id not in API_MESSAGES:
        raise KeyError(f"Message ID '{message_id}' not found")
    
    return API_MESSAGES[message_id]

def verify_message_signature(message_content, signature):
    """
    Verifies if the signature matches the hash of the message content.
    
    Args:
        message_content (str): The message content to verify
        signature (str): The signature to compare against
        
    Returns:
        bool: True if signature matches, False otherwise
    """
    try:
        # Ensure inputs are strings
        if not isinstance(message_content, str) or not isinstance(signature, str):
            return False
        
        # Compute SHA256 hash of the message content
        message_hash = hashlib.sha256(message_content.encode()).hexdigest()
        
        # Compare the computed hash with the provided signature
        return message_hash == signature
        
    except Exception:
        # Return False for any unsupported operations or errors
        return False
