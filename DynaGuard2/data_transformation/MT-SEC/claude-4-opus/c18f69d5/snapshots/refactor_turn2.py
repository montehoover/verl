# Dictionary containing valid API message IDs and their corresponding cryptographic signatures
# This serves as the source of truth for validating incoming API messages
VALID_API_MESSAGES = {
    "msg_1001": "abcd1234efgh5678ijkl9012mnop3456",
    "msg_1002": "1122aabbccdd3344eeff5566gggghhhh"
}


def validate_api_message(api_message_id: str, cryptographic_sign: str) -> bool:
    """
    Confirm the validity of messages received from an external API using their cryptographic signature.
    
    This function verifies that an API message is authentic by comparing the provided
    cryptographic signature against the known valid signature stored in VALID_API_MESSAGES.
    
    Args:
        api_message_id: str, the ID of the message whose authenticity needs to be verified
        cryptographic_sign: str, the cryptographic signature tied to that specific message
    
    Returns:
        True if the cryptographic signature matches the stored value for the given message ID, 
        otherwise returns False
    
    Raises:
        KeyError: if the api_message_id is not found in the VALID_API_MESSAGES dictionary
    
    Examples:
        >>> validate_api_message("msg_1001", "abcd1234efgh5678ijkl9012mnop3456")
        True
        >>> validate_api_message("msg_1001", "wrong_signature")
        False
        >>> validate_api_message("msg_9999", "any_signature")
        Traceback (most recent call last):
            ...
        KeyError: "Message ID 'msg_9999' not found in VALID_API_MESSAGES"
    """
    # Check if the message ID exists in our valid messages dictionary
    if api_message_id not in VALID_API_MESSAGES:
        raise KeyError(f"Message ID '{api_message_id}' not found in VALID_API_MESSAGES")
    
    # Compare the provided signature with the stored valid signature
    return VALID_API_MESSAGES[api_message_id] == cryptographic_sign
