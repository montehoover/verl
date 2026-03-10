"""Module for authenticating API messages using cryptographic signatures."""

import logging

# Configure logging for this module
logger = logging.getLogger(__name__)

# Dictionary mapping message IDs to their expected cryptographic signatures
VALID_API_MESSAGES = {
    "msg_1001": "abcd1234efgh5678ijkl9012mnop3456",
    "msg_1002": "1122aabbccdd3344eeff5566gggghhhh"
}


def authenticate_message(message_id: str, signature: str) -> bool:
    """
    Verify the authenticity of an API message using its cryptographic signature.
    
    This function checks if the provided signature matches the expected signature
    for a given message ID. It ensures that only messages with valid signatures
    are accepted from the external API.
    
    Args:
        message_id (str): The unique identifier of the message to be authenticated.
        signature (str): The cryptographic signature associated with the message.
    
    Returns:
        bool: True if the message signature is valid, False otherwise.
    
    Raises:
        KeyError: If the message_id is not found in the VALID_API_MESSAGES dictionary.
    """
    logger.info(f"Authentication attempt for message_id: {message_id}")
    
    # Check if the message ID exists in our registry
    if message_id not in VALID_API_MESSAGES:
        logger.error(f"Authentication failed - message_id not found: {message_id}")
        raise KeyError(f"Message ID '{message_id}' not found")
    
    # Verify the signature matches the expected value
    is_valid = VALID_API_MESSAGES[message_id] == signature
    
    if is_valid:
        logger.info(f"Authentication successful for message_id: {message_id}")
    else:
        logger.warning(f"Authentication failed - invalid signature for message_id: {message_id}")
    
    return is_valid
