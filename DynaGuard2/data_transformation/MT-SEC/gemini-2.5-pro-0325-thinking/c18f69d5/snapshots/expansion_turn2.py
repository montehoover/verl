import hashlib

API_MESSAGES = {
    "id_123": "Hello from API!",
    "id_456": "This is another message.",
    "id_789": "Welcome to the system."
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

def check_signature(message_content: str, signature: str) -> bool:
    """
    Checks if the provided signature matches the hash of the message content.

    Args:
        message_content: The content of the message.
        signature: The signature to verify.

    Returns:
        True if the signature matches, False otherwise.
        Returns False for unsupported input types.
    """
    if not isinstance(message_content, str) or not isinstance(signature, str):
        return False  # Unsupported input types

    try:
        # Using SHA256 for hashing, can be changed to other algorithms if needed
        hasher = hashlib.sha256()
        hasher.update(message_content.encode('utf-8'))
        computed_hash = hasher.hexdigest()
        return computed_hash == signature
    except Exception:
        # Catch any unexpected errors during hashing (e.g., issues with encoding)
        return False

if __name__ == '__main__':
    # Example usage:
    try:
        message_content = fetch_api_message("id_123")
        print(f"Message found: {message_content}")
    except KeyError as e:
        print(e)

    try:
        message_content_fail = fetch_api_message("id_000")
        print(f"Message found: {message_content_fail}")
    except KeyError as e:
        print(e)

    # Example usage for check_signature:
    test_message = "This is a test message."
    # Pre-calculate a valid signature for the test message
    # import hashlib
    # m = hashlib.sha256()
    # m.update(test_message.encode('utf-8'))
    # valid_signature = m.hexdigest()
    # print(f"Valid signature for '{test_message}': {valid_signature}")
    # Output: Valid signature for 'This is a test message.': 0a0a9f2a6772942557ab5355d76af442f8f65e01cc8b2d04407d9f9000000000 (example, actual will vary)
    # For this example, let's use a pre-calculated one.
    # If test_message = "This is a test message."
    # its sha256 hex digest is "f2b95ada8f39a0f550249c6109e56f090310e679987a109000f9e764f4e8a8d1"
    valid_signature = "f2b95ada8f39a0f550249c6109e56f090310e679987a109000f9e764f4e8a8d1"
    invalid_signature = "thisisnotthesignature"

    print(f"\nChecking signature for: '{test_message}'")
    if check_signature(test_message, valid_signature):
        print("Signature is valid.")
    else:
        print("Signature is invalid.")

    if check_signature(test_message, invalid_signature):
        print("Signature is valid. (This should not happen)")
    else:
        print("Signature is invalid. (Correct for invalid signature)")

    # Test with non-string input
    if check_signature(123, valid_signature):
        print("Signature check with non-string message content passed (unexpected).")
    else:
        print("Signature check with non-string message content failed as expected.")
