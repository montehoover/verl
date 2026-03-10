from hmac import compare_digest

VALID_API_MESSAGES = {
    "msg_1001": "abcd1234efgh5678ijkl9012mnop3456",
    "msg_1002": "1122aabbccdd3344eeff5566gggghhhh"
}


def verify_message(api_message_id: str, crypto_signature: str) -> bool:
    """
    Verify the legitimacy of a message from an external API using its cryptographic signature.

    Args:
        api_message_id: Unique identifier for the message being checked.
        crypto_signature: The cryptographic signature to validate against the stored signature.

    Returns:
        True if the cryptographic signature is valid for the given api_message_id, else False.

    Raises:
        KeyError: If the given api_message_id does not exist in VALID_API_MESSAGES.
    """
    expected_signature = VALID_API_MESSAGES[api_message_id]
    return compare_digest(expected_signature, crypto_signature)
