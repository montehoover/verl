import hmac

VALID_API_MESSAGES = {
    "msg_1001": "abcd1234efgh5678ijkl9012mnop3456",
    "msg_1002": "1122aabbccdd3344eeff5566gggghhhh"
}

def validate_api_message(api_message_id: str, cryptographic_sign: str) -> bool:
    """
    Confirm the validity of a message received from an external API by verifying its cryptographic signature.

    Args:
        api_message_id: The ID of the message whose authenticity needs to be verified.
        cryptographic_sign: The cryptographic signature to validate.

    Returns:
        True if the cryptographic signature matches the stored value for the given message ID, otherwise False.

    Raises:
        KeyError: If the api_message_id is not found in VALID_API_MESSAGES.
    """
    expected_signature = VALID_API_MESSAGES[api_message_id]  # may raise KeyError if not found
    return hmac.compare_digest(expected_signature, cryptographic_sign)
