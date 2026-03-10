import hashlib

# Example API messages dictionary
API_MESSAGES = {
    "msg_001": "Welcome to our service!",
    "msg_002": "Your request has been processed successfully.",
    "msg_003": "An error occurred. Please try again later.",
    "msg_004": "Thank you for using our API.",
    "msg_005": "Invalid input provided. Please check your data."
}

# Valid API messages with their expected cryptographic signatures
VALID_API_MESSAGES = {
    "msg_1001": "abcd1234efgh5678ijkl9012mnop3456",
    "msg_1002": "1122aabbccdd3344eeff5566gggghhhh"
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

def verify_message(api_message_id, crypto_signature):
    """
    Verifies the legitimacy of a message from an external API using its cryptographic signature.
    
    Args:
        api_message_id: The ID of the API message to verify
        crypto_signature: The cryptographic signature to validate against
        
    Returns:
        True if the cryptographic signature is valid for the given api_message_id, else False
    """
    # Check if the api_message_id exists in VALID_API_MESSAGES
    if api_message_id not in VALID_API_MESSAGES:
        return False
    
    # Get the expected signature for this message ID
    expected_signature = VALID_API_MESSAGES[api_message_id]
    
    # Compare the provided signature with the expected signature
    return crypto_signature == expected_signature
