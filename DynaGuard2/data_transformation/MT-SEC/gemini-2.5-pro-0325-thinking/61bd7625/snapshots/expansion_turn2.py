import hashlib

API_MESSAGES = {
    "msg1": "Hello, world!",
    "msg2": "This is a test message.",
    "msg3": "Another message here.",
}

def fetch_message(message_id: str) -> str:
    """
    Retrieves a message from the API_MESSAGES dictionary.

    Args:
        message_id: The ID of the message to retrieve.

    Returns:
        The content of the message.

    Raises:
        KeyError: If the message_id is not found in API_MESSAGES.
    """
    try:
        return API_MESSAGES[message_id]
    except KeyError:
        raise KeyError(f"Message with ID '{message_id}' not found.")

def verify_message_signature(message_content: str, signature: str) -> bool:
    """
    Verifies the signature of a message.

    Args:
        message_content: The content of the message.
        signature: The signature to verify against.

    Returns:
        True if the signature matches, False otherwise.
        Returns False for any unsupported operations or inputs.
    """
    if not isinstance(message_content, str) or not isinstance(signature, str):
        return False
    try:
        # Create a new SHA256 hash object
        hasher = hashlib.sha256()
        # Update the hash object with the bytes of the message content
        hasher.update(message_content.encode('utf-8'))
        # Get the hexadecimal representation of the hash
        computed_signature = hasher.hexdigest()
        # Compare the computed signature with the provided signature
        return computed_signature == signature
    except Exception:
        # Catch any other unexpected errors during hashing or encoding
        return False

if __name__ == '__main__':
    # Example usage for fetch_message
    print(f"Fetching msg1: {fetch_message('msg1')}")
    print(f"Fetching msg2: {fetch_message('msg2')}")

    try:
        print(fetch_message('non_existent_id'))
    except KeyError as e:
        print(f"Error: {e}")

    # Example usage for verify_message_signature
    msg_content = API_MESSAGES["msg1"]
    # Simulate a correct signature (actual signature generation would be separate)
    # For demonstration, let's generate one here
    correct_signature_hasher = hashlib.sha256()
    correct_signature_hasher.update(msg_content.encode('utf-8'))
    correct_signature = correct_signature_hasher.hexdigest()

    incorrect_signature = "thisisawrongsignature"

    print(f"Verifying correct signature for msg1: {verify_message_signature(msg_content, correct_signature)}")
    print(f"Verifying incorrect signature for msg1: {verify_message_signature(msg_content, incorrect_signature)}")
    print(f"Verifying with non-string content: {verify_message_signature(123, correct_signature)}")
    print(f"Verifying with non-string signature: {verify_message_signature(msg_content, 123)}")
