import hmac
from typing import Dict

# Predefined global dictionary holding the correct cryptographic signatures
VALID_API_MESSAGES: Dict[str, str] = {
    "msg_1001": "abcd1234efgh5678ijkl9012mnop3456",
    "msg_1002": "1122aabbccdd3344eeff5566gggghhhh"
}

def validate_api_message(api_message_id: str, cryptographic_sign: str) -> bool:
    """
    Confirm the validity of a message received from an external API using its cryptographic signature.

    Args:
        api_message_id: The ID of the message whose authenticity needs to be verified.
        cryptographic_sign: The cryptographic signature tied to that specific message.

    Returns:
        True if the cryptographic signature matches the stored value for the given message ID, otherwise False.

    Raises:
        KeyError: If the api_message_id is not found in the VALID_API_MESSAGES dictionary.
    """
    # Attempt to retrieve the expected signature; this will raise KeyError if not found.
    expected_signature = VALID_API_MESSAGES[api_message_id]

    # Use constant-time comparison to mitigate timing attacks.
    return hmac.compare_digest(str(cryptographic_sign), str(expected_signature))
