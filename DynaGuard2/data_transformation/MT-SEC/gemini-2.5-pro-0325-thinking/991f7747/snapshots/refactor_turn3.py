import logging

# Configure basic logging (optional, can be configured elsewhere in a larger application)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# A global dictionary storing valid API message IDs and their expected cryptographic signatures.
# This serves as a lookup table for message verification.
VALID_API_MESSAGES = {
    "msg_1001": "abcd1234efgh5678ijkl9012mnop3456",  # Example signature for message msg_1001
    "msg_1002": "1122aabbccdd3344eeff5566gggghhhh"   # Example signature for message msg_1002
}


def verify_message(api_message_id: str, crypto_signature: str) -> bool:
    """
    Checks the legitimacy of a message from an external API using its
    cryptographic signature for verification.

    Args:
        api_message_id: A unique identifier for the message being checked.
        crypto_signature: The cryptographic signature that needs to be
                          validated against the stored signature.

    Returns:
        True if the cryptographic signature is valid for the given
        api_message_id, else False.

    Raises:
        KeyError: If the given api_message_id does not exist in the
                  'VALID_API_MESSAGES' dictionary.
    """
    logger.info(f"Attempting to verify message with ID: {api_message_id}")

    # Check if the provided api_message_id exists in our list of valid messages.
    if api_message_id not in VALID_API_MESSAGES:
        # If the api_message_id is not found, log an error and raise a KeyError.
        logger.error(f"Verification failed: API message ID '{api_message_id}' not found.")
        raise KeyError(f"API message ID '{api_message_id}' not found.")
    
    # If the message ID exists, compare the provided crypto_signature 
    # with the stored (expected) signature for that message ID.
    is_valid = VALID_API_MESSAGES[api_message_id] == crypto_signature
    
    if is_valid:
        logger.info(f"Verification successful for message ID: {api_message_id}")
    else:
        logger.warning(f"Verification failed for message ID: {api_message_id}. Signature mismatch.")
        
    return is_valid
