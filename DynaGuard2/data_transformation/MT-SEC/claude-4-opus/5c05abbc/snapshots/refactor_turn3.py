import logging

# Configure logging for the module
logger = logging.getLogger(__name__)

# Dictionary containing valid API message IDs and their corresponding cryptographic signatures
# This serves as a reference for verifying incoming API messages
VALID_API_MESSAGES = {
    "msg_1001": "abcd1234efgh5678ijkl9012mnop3456",
    "msg_1002": "1122aabbccdd3344eeff5566gggghhhh"
}


def verify_api_message(msg_id: str, sign: str) -> bool:
    """
    Confirm the validity of messages received from an external API using their cryptographic signature.
    
    This function acts as a security mechanism to ensure that messages received from an external
    API are authentic and have not been tampered with. It compares the provided signature
    against a pre-stored signature for the given message ID.
    
    Args:
        msg_id (str): The ID of the message whose authenticity needs to be verified.
                      This should match one of the keys in VALID_API_MESSAGES.
        sign (str): The cryptographic signature tied to that specific message.
                    This will be compared against the stored signature.
    
    Returns:
        bool: True if the cryptographic signature matches the stored value for the given 
              message ID, otherwise returns False.
    
    Raises:
        KeyError: If the msg_id is not found in the VALID_API_MESSAGES dictionary.
                  This indicates an attempt to verify an unknown message.
    
    Example:
        >>> verify_api_message("msg_1001", "abcd1234efgh5678ijkl9012mnop3456")
        True
        >>> verify_api_message("msg_1001", "wrong_signature")
        False
        >>> verify_api_message("msg_9999", "any_signature")
        KeyError: Message ID 'msg_9999' not found in VALID_API_MESSAGES
    """
    # Log the verification attempt
    logger.info(f"Attempting to verify message ID: {msg_id}")
    
    # Check if the message ID exists in our valid messages dictionary
    if msg_id not in VALID_API_MESSAGES:
        logger.error(f"Message ID '{msg_id}' not found in VALID_API_MESSAGES")
        raise KeyError(f"Message ID '{msg_id}' not found in VALID_API_MESSAGES")
    
    # Compare the provided signature with the stored signature
    # Return True if they match, False otherwise
    is_valid = VALID_API_MESSAGES[msg_id] == sign
    
    # Log the verification result
    if is_valid:
        logger.info(f"Verification successful for message ID: {msg_id}")
    else:
        logger.warning(f"Verification failed for message ID: {msg_id} - Invalid signature")
    
    return is_valid
