"""
Module for authenticating messages from an external API.

This module provides functionality to verify messages based on their
cryptographic signatures against a predefined set of valid messages.
"""
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# A dictionary mapping message IDs to their expected cryptographic signatures.
# This acts as a mock database of valid API messages and their signatures.
VALID_API_MESSAGES = {
    "msg_1001": "abcd1234efgh5678ijkl9012mnop3456",
    "msg_1002": "1122aabbccdd3344eeff5566gggghhhh"
}


def authenticate_message(message_id: str, signature: str) -> bool:
    """
    Verifies a message from an external API using a cryptographic signature.

    This function checks if the provided `message_id` exists in the
    `VALID_API_MESSAGES` store. If it exists, it compares the provided
    `signature` with the stored signature for that message ID.

    Args:
        message_id: The unique identifier of the message to be authenticated.
        signature: The cryptographic signature associated with the message.

    Returns:
        True if the message signature is valid (i.e., matches the expected
        signature for the given message_id), False otherwise.

    Raises:
        KeyError: If the `message_id` is not found in `VALID_API_MESSAGES`.
    """
    # Check if the message_id exists in our record of valid messages.
    if message_id not in VALID_API_MESSAGES:
        logger.error(f"Authentication attempt for non-existent message_id: {message_id}")
        raise KeyError(f"Message ID '{message_id}' not found.")
    
    # Retrieve the expected signature for the given message_id.
    expected_signature = VALID_API_MESSAGES[message_id]
    
    # Compare the provided signature with the expected signature.
    is_valid = signature == expected_signature
    
    if is_valid:
        logger.info(f"Authentication successful for message_id: {message_id}")
    else:
        logger.warning(f"Authentication failed for message_id: {message_id}")
        
    return is_valid
