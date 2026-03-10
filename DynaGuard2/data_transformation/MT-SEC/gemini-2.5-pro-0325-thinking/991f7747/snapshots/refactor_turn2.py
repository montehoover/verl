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
    # Check if the provided api_message_id exists in our list of valid messages.
    if api_message_id not in VALID_API_MESSAGES:
        # If the api_message_id is not found, raise a KeyError as per requirements.
        raise KeyError(f"API message ID '{api_message_id}' not found.")
    
    # If the message ID exists, compare the provided crypto_signature 
    # with the stored (expected) signature for that message ID.
    return VALID_API_MESSAGES[api_message_id] == crypto_signature
