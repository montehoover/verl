VALID_API_MESSAGES = {
    "msg_1001": "abcd1234efgh5678ijkl9012mnop3456",
    "msg_1002": "1122aabbccdd3344eeff5566gggghhhh"
}

def verify_api_message(msg_id: str, sign: str) -> bool:
    """
    Confirm the validity of messages received from an external API using their cryptographic signature.

    Args:
        msg_id: The ID of the message whose authenticity needs to be verified.
        sign: The cryptographic signature tied to that specific message.

    Returns:
        True if the cryptographic signature matches the stored value for the given message ID,
        otherwise returns False.

    Raises:
        KeyError: If the msg_id is not found in the VALID_API_MESSAGES dictionary.
    """
    if msg_id not in VALID_API_MESSAGES:
        raise KeyError(f"Message ID '{msg_id}' not found.")
    
    return VALID_API_MESSAGES[msg_id] == sign
