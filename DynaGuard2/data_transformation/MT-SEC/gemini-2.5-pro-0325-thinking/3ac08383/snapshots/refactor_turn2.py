VALID_API_MESSAGES = {
    "msg_1001": "abcd1234efgh5678ijkl9012mnop3456",
    "msg_1002": "1122aabbccdd3344eeff5566gggghhhh"
}


def _ensure_message_exists(api_message: str, valid_messages: dict) -> None:
    """
    Ensures the API message exists in the valid messages dictionary.

    Args:
        api_message: The API message to check.
        valid_messages: The dictionary of valid API messages and their signatures.

    Raises:
        KeyError: If the api_message is not found in valid_messages.
    """
    if api_message not in valid_messages:
        raise KeyError(f"API message '{api_message}' not found.")


def _is_signature_valid(expected_sig: str, provided_sig: str) -> bool:
    """
    Compares the expected signature with the provided signature.

    Args:
        expected_sig: The expected cryptographic signature.
        provided_sig: The cryptographic signature provided with the message.

    Returns:
        True if the signatures match, False otherwise.
    """
    return expected_sig == provided_sig


def check_api_signature(api_message: str, sig: str) -> bool:
    """
    Checks the legitimacy of a message from an external API using its cryptographic signature.

    Args:
        api_message: A unique identifier for the message being checked.
        sig: The cryptographic signature that needs to be validated.

    Returns:
        True if the cryptographic signature is valid, False otherwise.

    Raises:
        KeyError: If the given api_message does not exist in VALID_API_MESSAGES.
    """
    _ensure_message_exists(api_message, VALID_API_MESSAGES)
    
    expected_sig = VALID_API_MESSAGES[api_message]
    return _is_signature_valid(expected_sig, sig)
