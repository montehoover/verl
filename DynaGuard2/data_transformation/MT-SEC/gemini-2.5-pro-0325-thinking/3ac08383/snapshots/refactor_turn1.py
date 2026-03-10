VALID_API_MESSAGES = {
    "msg_1001": "abcd1234efgh5678ijkl9012mnop3456",
    "msg_1002": "1122aabbccdd3344eeff5566gggghhhh"
}

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
    if api_message not in VALID_API_MESSAGES:
        raise KeyError(f"API message '{api_message}' not found.")
    
    expected_sig = VALID_API_MESSAGES[api_message]
    return sig == expected_sig
