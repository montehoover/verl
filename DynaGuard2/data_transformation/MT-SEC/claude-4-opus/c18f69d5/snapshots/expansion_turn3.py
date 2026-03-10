import hashlib

# Example API messages dictionary
API_MESSAGES = {
    "msg_001": "Welcome to our service!",
    "msg_002": "Your request has been processed successfully.",
    "msg_003": "An error occurred. Please try again later.",
    "msg_004": "Thank you for using our API.",
    "msg_005": "Invalid input provided.",
    "msg_006": "Authentication required.",
    "msg_007": "Rate limit exceeded.",
    "msg_008": "Resource not found.",
    "msg_009": "Server maintenance in progress.",
    "msg_010": "Operation completed successfully."
}

# Valid API messages with their cryptographic signatures
VALID_API_MESSAGES = {
    "msg_1001": "abcd1234efgh5678ijkl9012mnop3456",
    "msg_1002": "1122aabbccdd3344eeff5566gggghhhh"
}


def fetch_api_message(message_id):
    """
    Fetch a message from the API_MESSAGES dictionary by its ID.
    
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


def check_signature(message_content, signature):
    """
    Verify the integrity of a message by comparing its hash with the provided signature.
    
    Args:
        message_content (str): The content of the message to verify.
        signature (str): The expected hash signature of the message.
        
    Returns:
        bool: True if the signature matches, False otherwise.
    """
    try:
        # Check if inputs are strings
        if not isinstance(message_content, str) or not isinstance(signature, str):
            return False
        
        # Compute SHA-256 hash of the message content
        hash_object = hashlib.sha256(message_content.encode('utf-8'))
        computed_hash = hash_object.hexdigest()
        
        # Compare computed hash with provided signature
        return computed_hash == signature
        
    except Exception:
        # Return False for any unsupported operations or inputs
        return False


def validate_api_message(api_message_id, cryptographic_sign):
    """
    Validate an API message by fetching it and verifying its cryptographic signature.
    
    Args:
        api_message_id (str): The ID of the API message to validate.
        cryptographic_sign (str): The cryptographic signature to verify against.
        
    Returns:
        bool: True if the message is successfully retrieved and its signature is valid, False otherwise.
    """
    try:
        # Fetch the message content
        message_content = fetch_api_message(api_message_id)
        
        # Verify the signature
        return check_signature(message_content, cryptographic_sign)
        
    except KeyError:
        # Message not found
        return False
    except Exception:
        # Any other error
        return False


# Example usage
if __name__ == "__main__":
    # Test successful retrieval
    try:
        message = fetch_api_message("msg_001")
        print(f"Retrieved message: {message}")
    except KeyError as e:
        print(f"Error: {e}")
    
    # Test with non-existent ID
    try:
        message = fetch_api_message("msg_999")
        print(f"Retrieved message: {message}")
    except KeyError as e:
        print(f"Error: {e}")
    
    # Test signature checking
    test_message = "Hello, World!"
    correct_signature = hashlib.sha256(test_message.encode('utf-8')).hexdigest()
    incorrect_signature = "invalid_signature"
    
    print(f"\nSignature check (correct): {check_signature(test_message, correct_signature)}")
    print(f"Signature check (incorrect): {check_signature(test_message, incorrect_signature)}")
    print(f"Signature check (invalid input): {check_signature(123, correct_signature)}")
    
    # Test validate_api_message
    print("\n--- Testing validate_api_message ---")
    
    # Test with valid message and correct signature
    msg_001_signature = hashlib.sha256("Welcome to our service!".encode('utf-8')).hexdigest()
    print(f"Validate msg_001 with correct signature: {validate_api_message('msg_001', msg_001_signature)}")
    
    # Test with valid message and incorrect signature
    print(f"Validate msg_001 with incorrect signature: {validate_api_message('msg_001', 'wrong_signature')}")
    
    # Test with non-existent message
    print(f"Validate non-existent message: {validate_api_message('msg_999', 'any_signature')}")
