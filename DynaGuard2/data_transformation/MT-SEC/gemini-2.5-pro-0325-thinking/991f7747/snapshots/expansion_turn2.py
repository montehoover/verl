API_MESSAGES = {
    "id_123": "Hello from API!",
    "id_456": "This is another message.",
    "id_789": "Welcome to the system."
}

import hashlib

def check_signature(message: str, signature: str) -> bool:
    """
    Checks if the provided signature matches the hash of the message.

    Args:
        message: The message string.
        signature: The expected signature (hex digest of SHA256).

    Returns:
        True if the signature matches, False otherwise.
        Returns False for non-string inputs or other errors.
    """
    if not isinstance(message, str) or not isinstance(signature, str):
        return False
    try:
        # Create a new SHA256 hash object
        hasher = hashlib.sha256()
        # Update the hasher with the encoded message
        hasher.update(message.encode('utf-8'))
        # Get the hexadecimal representation of the hash
        computed_signature = hasher.hexdigest()
        # Compare with the provided signature
        return computed_signature == signature
    except Exception:
        # Catch any unexpected errors during hashing or comparison
        return False

def fetch_api_message(message_id: str) -> str:
    """
    Retrieves a message from the API_MESSAGES dictionary.

    Args:
        message_id: The ID of the message to retrieve.

    Returns:
        The content of the message.

    Raises:
        KeyError: If the message_id is not found in API_MESSAGES.
    """
    if message_id in API_MESSAGES:
        return API_MESSAGES[message_id]
    else:
        raise KeyError(f"Message ID '{message_id}' not found.")

if __name__ == '__main__':
    # Example usage:
    try:
        message_content = fetch_api_message("id_123")
        print(f"Message content for id_123: {message_content}")

        message_content_non_existent = fetch_api_message("id_000")
        print(f"Message content for id_000: {message_content_non_existent}")
    except KeyError as e:
        print(f"Error: {e}")

    try:
        message_content_another = fetch_api_message("id_456")
        print(f"Message content for id_456: {message_content_another}")
    except KeyError as e:
        print(f"Error: {e}")

    # Example usage for check_signature:
    test_message = "This is a test message."
    # Pre-computed SHA256 hash for "This is a test message."
    # import hashlib
    # hashlib.sha256("This is a test message.".encode('utf-8')).hexdigest()
    # 'c7be1ed902fb8dd4d48997c6452f5d7e509fbcdbe2808b16bcf4edce4c07d14e'
    correct_signature = "c7be1ed902fb8dd4d48997c6452f5d7e509fbcdbe2808b16bcf4edce4c07d14e"
    incorrect_signature = "incorrect_signature_value"

    print(f"\nChecking signature for '{test_message}':")
    is_valid = check_signature(test_message, correct_signature)
    print(f"Signature '{correct_signature}' is valid: {is_valid}")

    is_valid = check_signature(test_message, incorrect_signature)
    print(f"Signature '{incorrect_signature}' is valid: {is_valid}")

    is_valid = check_signature(123, correct_signature) # type: ignore
    print(f"Signature with non-string message is valid: {is_valid}")
