import hashlib

API_MESSAGES = {
    "id_123": "Hello, this is a message from the API.",
    "id_456": "Another message here.",
    "id_789": "A final test message.",
}

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
        # Compare the computed signature with the provided signature
        return computed_signature == signature
    except Exception:
        # Catch any unexpected errors during hashing or comparison
        return False

if __name__ == '__main__':
    # Example usage for fetch_api_message:
    try:
        message_content = fetch_api_message("id_123")
        print(f"Message content for id_123: {message_content}")

        message_content_non_existent = fetch_api_message("id_000")
        print(f"Message content for id_000: {message_content_non_existent}")
    except KeyError as e:
        print(f"Error: {e}")

    try:
        message_content_2 = fetch_api_message("id_456")
        print(f"Message content for id_456: {message_content_2}")
    except KeyError as e:
        print(f"Error: {e}")

    # Example usage for check_signature:
    test_message = "This is a test message."
    # Pre-calculate a valid signature for the test message
    # import hashlib
    # hashlib.sha256("This is a test message.".encode('utf-8')).hexdigest()
    # Output: 'c1591350f8736972cc90f68a49a730dc011b06999876f2a1011c1792f9199879'
    valid_signature = "c1591350f8736972cc90f68a49a730dc011b06999876f2a1011c1792f9199879"
    invalid_signature = "invalidsignature123"

    print(f"\nChecking signature for '{test_message}':")
    is_valid = check_signature(test_message, valid_signature)
    print(f"Signature '{valid_signature}' is valid: {is_valid}")

    is_valid_again = check_signature(test_message, invalid_signature)
    print(f"Signature '{invalid_signature}' is valid: {is_valid_again}")

    # Test with non-string input
    is_valid_non_string = check_signature(123, valid_signature)
    print(f"Signature check with non-string message: {is_valid_non_string}")

    is_valid_non_string_sig = check_signature(test_message, None)
    print(f"Signature check with non-string signature: {is_valid_non_string_sig}")
