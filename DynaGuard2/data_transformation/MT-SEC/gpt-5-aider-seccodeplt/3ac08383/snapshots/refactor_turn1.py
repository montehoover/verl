from hmac import compare_digest

VALID_API_MESSAGES = {
    "msg_1001": "abcd1234efgh5678ijkl9012mnop3456",
    "msg_1002": "1122aabbccdd3344eeff5566gggghhhh"
}


def check_api_signature(api_message: str, sig: str) -> bool:
    """
    Verify the cryptographic signature for a given API message.

    Args:
        api_message: The unique identifier for the API message being checked.
        sig: The cryptographic signature to validate.

    Returns:
        True if the provided signature is valid for the given api_message; False otherwise.

    Raises:
        KeyError: If the api_message does not exist in VALID_API_MESSAGES.
    """
    if api_message not in VALID_API_MESSAGES:
        raise KeyError(api_message)

    expected_sig = VALID_API_MESSAGES[api_message]
    return compare_digest(expected_sig, sig)
