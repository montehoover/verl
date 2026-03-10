VALID_API_MESSAGES = {
    "msg_1001": "abcd1234efgh5678ijkl9012mnop3456",
    "msg_1002": "1122aabbccdd3344eeff5566gggghhhh"
}

def verify_message(api_message_id: str, crypto_signature: str) -> bool:
    """
    Checks the legitimacy of a message from an external API using its cryptographic signature.

    Args:
        api_message_id: A unique identifier for the message being checked.
        crypto_signature: The cryptographic signature that needs to be validated.

    Returns:
        True if the cryptographic signature is valid, False otherwise.

    Raises:
        KeyError: If the given api_message_id does not exist in VALID_API_MESSAGES.
    """
    if api_message_id not in VALID_API_MESSAGES:
        raise KeyError(f"API message ID '{api_message_id}' not found.")
    
    expected_signature = VALID_API_MESSAGES[api_message_id]
    return crypto_signature == expected_signature
