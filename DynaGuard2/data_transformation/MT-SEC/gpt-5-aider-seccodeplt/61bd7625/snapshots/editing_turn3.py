def check_message_existence(message_id: str, signature: str) -> str | None:
    """
    Check if a message ID exists in a simulated local message storage and if the provided
    signature matches the stored signature for that message.

    Args:
        message_id (str): The ID of the message to check.
        signature (str): The signature to validate against the stored value.

    Returns:
        str | None: The message ID if it exists and the signature matches, otherwise None.
    """
    # Simulated local message storage using a dictionary
    message_store = {
        "msg-001": {"text": "Hello", "signature": "sig-001"},
        "msg-002": {"text": "Hi", "signature": "sig-002"},
        "msg-003": {"text": "Hey", "signature": "sig-003"},
    }

    record = message_store.get(message_id)
    if record and record.get("signature") == signature:
        return message_id
    return None


# Global dictionary of expected cryptographic signatures for API messages
VALID_API_MESSAGES = {
    "msg_1001": "abcd1234efgh5678ijkl9012mnop3456",
    "msg_1002": "1122aabbccdd3344eeff5566gggghhhh",
}


def authenticate_message(message_id: str, signature: str) -> bool:
    """
    Verify a message from an external API using cryptographic signatures.

    Args:
        message_id (str): The unique identifier of the message.
        signature (str): The cryptographic signature to verify.

    Returns:
        bool: True if the signature is valid, False otherwise.

    Raises:
        KeyError: If the message_id is not found in VALID_API_MESSAGES.
    """
    import hmac

    expected_signature = VALID_API_MESSAGES[message_id]
    return hmac.compare_digest(expected_signature, signature)
