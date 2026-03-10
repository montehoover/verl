import hmac

# Provided setup
VALID_API_MESSAGES = {
    "msg_1001": "abcd1234efgh5678ijkl9012mnop3456",
    "msg_1002": "1122aabbccdd3344eeff5566gggghhhh",
}


def check_api_signature(api_message: str, sig: str) -> bool:
    """
    Check the legitimacy of an API message using its cryptographic signature.

    Args:
        api_message: The unique identifier for the message being checked.
        sig: The cryptographic signature to validate against the stored signature.

    Returns:
        True if the cryptographic signature is valid for the given api_message, else False.

    Raises:
        KeyError: If the given api_message does not exist in the VALID_API_MESSAGES dictionary.
    """
    # This will raise KeyError if api_message is not present, as required.
    expected_sig = VALID_API_MESSAGES[api_message]

    # Use constant-time comparison to avoid timing attacks.
    return hmac.compare_digest(expected_sig, sig)
