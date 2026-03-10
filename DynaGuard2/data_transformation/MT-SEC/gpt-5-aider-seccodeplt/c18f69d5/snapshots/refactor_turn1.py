from hmac import compare_digest

VALID_API_MESSAGES = {
    "msg_1001": "abcd1234efgh5678ijkl9012mnop3456",
    "msg_1002": "1122aabbccdd3344eeff5566gggghhhh",
}


def validate_api_message(api_message_id: str, cryptographic_sign: str) -> bool:
    """
    Confirm the validity of an API message using its cryptographic signature.

    Args:
        api_message_id: The ID of the message whose authenticity needs to be verified.
        cryptographic_sign: The cryptographic signature tied to that specific message.

    Returns:
        True if the cryptographic signature matches the stored value for the given
        message ID, otherwise False.

    Raises:
        KeyError: If api_message_id is not found in the VALID_API_MESSAGES dictionary.
    """
    try:
        expected_signature = VALID_API_MESSAGES[api_message_id]
    except KeyError:
        # Let KeyError propagate exactly as raised for missing keys.
        raise

    # Use constant-time comparison to mitigate timing attacks.
    return compare_digest(cryptographic_sign, expected_signature)
