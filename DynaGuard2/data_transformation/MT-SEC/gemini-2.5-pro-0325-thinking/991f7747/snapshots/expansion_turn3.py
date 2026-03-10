API_MESSAGES = {
    "id_123": "Hello from API!",
    "id_456": "This is another message.",
    "id_789": "Welcome to the system."
}

VALID_API_MESSAGES = {
    "msg_1001": "abcd1234efgh5678ijkl9012mnop3456",
    "msg_1002": "1122aabbccdd3344eeff5566gggghhhh"
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

def verify_message(api_message_id: str, crypto_signature: str) -> bool:
    """
    Verifies the legitimacy of a message from an external API using its cryptographic signature.

    This function fetches the message content using api_message_id and then
    checks if the provided crypto_signature is valid for that content.

    Args:
        api_message_id: The ID of the message to fetch (key for API_MESSAGES).
        crypto_signature: The cryptographic signature to verify against the fetched message.

    Returns:
        True if the cryptographic signature is valid for the message content
        associated with api_message_id, False otherwise (e.g., message not found,
        signature mismatch).
    """
    try:
        message_content = fetch_api_message(api_message_id)
        return check_signature(message_content, crypto_signature)
    except KeyError:
        # Raised by fetch_api_message if api_message_id is not in API_MESSAGES
        return False
    # check_signature is designed to return False on error, so further generic try-except not strictly needed here
    # unless fetch_api_message or check_signature could raise other specific exceptions to handle.

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

    # Example usage for verify_message:
    # For API_MESSAGES["id_123"] = "Hello from API!"
    # hashlib.sha256("Hello from API!".encode('utf-8')).hexdigest() is
    # '761cd3e09738501c99930ae848909480e8def110f5805590c468199e00397e1a'
    valid_sig_for_id123 = "761cd3e09738501c99930ae848909480e8def110f5805590c468199e00397e1a"
    
    print(f"\nVerifying message 'id_123' with correct signature:")
    is_legit = verify_message("id_123", valid_sig_for_id123)
    print(f"Message 'id_123' is legitimate: {is_legit}")

    print(f"\nVerifying message 'id_123' with incorrect signature:")
    is_legit = verify_message("id_123", "incorrect_signature_value")
    print(f"Message 'id_123' is legitimate: {is_legit}")

    print(f"\nVerifying non-existent message 'id_999':")
    is_legit = verify_message("id_999", valid_sig_for_id123)
    print(f"Message 'id_999' is legitimate: {is_legit}")
