import hmac
from typing import Dict

VALID_API_MESSAGES: Dict[str, str] = {
    "msg_1001": "abcd1234efgh5678ijkl9012mnop3456",
    "msg_1002": "1122aabbccdd3344eeff5566gggghhhh"
}


def authenticate_message(message_id: str, signature: str) -> bool:
    """
    Verify a message's cryptographic signature against the expected value.

    Args:
        message_id: The unique identifier of the message to be authenticated.
        signature: The cryptographic signature associated with the message.

    Returns:
        True if the signature is valid for the given message_id, False otherwise.

    Raises:
        KeyError: If the message_id is not found in VALID_API_MESSAGES.
    """
    try:
        expected_signature = VALID_API_MESSAGES[message_id]
    except KeyError:
        # Propagate KeyError as specified
        raise

    # Use constant-time comparison to avoid timing attacks
    return hmac.compare_digest(signature, expected_signature)
