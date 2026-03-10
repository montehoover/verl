import logging

# Configure logging
logger = logging.getLogger(__name__)

# Dictionary containing valid API messages and their expected cryptographic signatures
VALID_API_MESSAGES = {
    "msg_1001": "abcd1234efgh5678ijkl9012mnop3456",
    "msg_1002": "1122aabbccdd3344eeff5566gggghhhh"
}


def verify_message(api_message_id: str, crypto_signature: str) -> bool:
    """
    Verify the legitimacy of a message from an external API using its cryptographic signature.
    
    This function checks if the provided cryptographic signature matches the expected
    signature stored for the given API message ID.
    
    Args:
        api_message_id (str): A unique identifier for the message being checked.
        crypto_signature (str): The cryptographic signature that needs to be validated
                                against the stored signature.
    
    Returns:
        bool: True if the cryptographic signature is valid for the given api_message_id,
              False otherwise.
    
    Raises:
        KeyError: If the given api_message_id does not exist in the VALID_API_MESSAGES
                  dictionary.
    """
    logger.info(f"Attempting to verify message with ID: {api_message_id}")
    
    # Check if the message ID exists in our valid messages
    if api_message_id not in VALID_API_MESSAGES:
        logger.error(f"Verification failed: api_message_id '{api_message_id}' does not exist")
        raise KeyError(f"api_message_id '{api_message_id}' does not exist")
    
    # Compare the provided signature with the expected signature
    is_valid = VALID_API_MESSAGES[api_message_id] == crypto_signature
    
    if is_valid:
        logger.info(f"Message verification successful for ID: {api_message_id}")
    else:
        logger.warning(f"Message verification failed for ID: {api_message_id} - Invalid signature")
    
    return is_valid
