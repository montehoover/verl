VALID_API_MESSAGES = {
    "msg_1001": "abcd1234efgh5678ijkl9012mnop3456",
    "msg_1002": "1122aabbccdd3344eeff5566gggghhhh"
}

def verify_api_message(msg_id: str, sign: str) -> bool:
    """
    Confirms the validity of messages received from an external API
    using their cryptographic signature.

    Args:
        msg_id: The ID of the message whose authenticity needs to be verified.
        sign: The cryptographic signature.

    Returns:
        True if the cryptographic signature matches the stored value
        for the given message ID, otherwise False.
    """
    return msg_id in VALID_API_MESSAGES and VALID_API_MESSAGES[msg_id] == sign
