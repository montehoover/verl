import logging

# Configure basic logging
# In a real application, this would likely be configured at the application entry point.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# A predefined global dictionary holding the correct cryptographic signatures
# for various API messages. Each key is a message ID, and the value is its
# corresponding cryptographic signature.
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
    logger.info(f"Attempting to validate API message with ID: {api_message_id}")

    # Check if the provided message ID exists in our record of valid messages.
    if api_message_id not in VALID_API_MESSAGES:
        # If the message ID is not found, log an error and raise a KeyError.
        error_message = f"API message ID '{api_message_id}' not found in VALID_API_MESSAGES."
        logger.error(error_message)
        raise KeyError(error_message)
    
    # Compare the provided cryptographic signature with the stored signature for the given message ID.
    expected_sign = VALID_API_MESSAGES[api_message_id]
    is_valid = expected_sign == cryptographic_sign

    if is_valid:
        logger.info(f"Validation successful for API message ID: {api_message_id}")
    else:
        logger.warning(
            f"Validation FAILED for API message ID: {api_message_id}. "
            f"Provided signature: '{cryptographic_sign}', Expected signature: '{expected_sign}'"
        )
        
    return is_valid
