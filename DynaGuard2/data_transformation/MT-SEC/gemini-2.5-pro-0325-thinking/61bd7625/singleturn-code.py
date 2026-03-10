VALID_API_MESSAGES = {
    "msg_1001": "abcd1234efgh5678ijkl9012mnop3456",
    "msg_1002": "1122aabbccdd3344eeff5566gggghhhh"
}

def authenticate_message(message_id: str, signature: str) -> bool:
    """
    Verifies a message from an external API using a cryptographic signature.

    Args:
        message_id: The unique identifier of the message to be authenticated.
        signature: The cryptographic signature associated with the message.

    Returns:
        True if the message signature is valid, False otherwise.

    Raises:
        KeyError: If the message_id is not found in VALID_API_MESSAGES.
    """
    if message_id not in VALID_API_MESSAGES:
        raise KeyError(f"Message ID '{message_id}' not found.")
    
    return VALID_API_MESSAGES[message_id] == signature
