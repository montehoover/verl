from hmac import compare_digest

# Global dictionary of expected signatures for API messages
VALID_API_MESSAGES = {
    "msg_1001": "abcd1234efgh5678ijkl9012mnop3456",
    "msg_1002": "1122aabbccdd3344eeff5566gggghhhh"
}

def authenticate_message(message_id: str, signature: str) -> bool:
    """
    Verify that the provided signature matches the expected signature for the given message_id.

    Args:
        message_id (str): The unique identifier of the message to be authenticated.
        signature (str): The cryptographic signature associated with the message.

    Returns:
        bool: True if the signature is valid for the message_id, False otherwise.

    Raises:
        KeyError: If the message_id is not found in the dictionary.
    """
    if message_id not in VALID_API_MESSAGES:
        raise KeyError(f"message_id not found: {message_id}")

    expected_signature = VALID_API_MESSAGES[message_id]
    # Use constant-time comparison to avoid timing attacks
    return compare_digest(signature, expected_signature)
