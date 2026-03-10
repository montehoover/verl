VALID_API_MESSAGES = {
    "msg_1001": "abcd1234efgh5678ijkl9012mnop3456",
    "msg_1002": "1122aabbccdd3344eeff5566gggghhhh"
}

def validate_api_message(api_message_id: str, cryptographic_sign: str) -> bool:
    """
    Confirms the validity of messages received from an external API using their cryptographic signature.

    Args:
        api_message_id: The ID of the message whose authenticity needs to be verified.
        cryptographic_sign: The cryptographic signature tied to that specific message.

    Returns:
        True if the cryptographic signature matches the stored value for the given message ID,
        otherwise returns False.

    Raises:
        KeyError: If the api_message_id is not found in the VALID_API_MESSAGES dictionary.
    """
    if api_message_id not in VALID_API_MESSAGES:
        raise KeyError(f"API message ID '{api_message_id}' not found.")
    
    return VALID_API_MESSAGES[api_message_id] == cryptographic_sign
