"""
Utilities for authenticating external API messages using constant-time
signature verification.

Exposes:
- VALID_API_MESSAGES: a mapping of message IDs to expected signatures.
- authenticate_message(): validates that a provided signature matches the
  expected value for a given message ID.

Security note:
- hmac.compare_digest is used to prevent timing attacks when comparing
  untrusted strings such as signatures.
"""

import hmac
from typing import Dict


# Mapping of known message IDs to their expected cryptographic signatures.
# In a real-world application, these values would typically be retrieved
# from a secure data store or configuration service rather than hard-coded.
VALID_API_MESSAGES: Dict[str, str] = {
    "msg_1001": "abcd1234efgh5678ijkl9012mnop3456",
    "msg_1002": "1122aabbccdd3344eeff5566gggghhhh",
}


def authenticate_message(message_id: str, signature: str) -> bool:
    """
    Verify that the provided signature matches the expected signature for
    the specified message ID.

    Args:
        message_id: The unique identifier of the message to be authenticated.
        signature: The cryptographic signature associated with the message.

    Returns:
        True if the provided signature matches the expected signature for
        the given message_id; False otherwise.

    Raises:
        KeyError: If the message_id is not found in VALID_API_MESSAGES.

    Example:
        >>> authenticate_message("msg_1001", "abcd1234efgh5678ijkl9012mnop3456")
        True
        >>> authenticate_message("msg_1001", "wrongsignature")
        False
        >>> authenticate_message("unknown_id", "anything")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        KeyError
    """
    # Retrieve the expected signature for the message_id. This will raise a
    # KeyError if the message_id does not exist, as specified by the contract.
    expected_signature = VALID_API_MESSAGES[message_id]

    # Compare the provided signature with the expected signature using a
    # constant-time comparison to mitigate timing side-channel attacks.
    return hmac.compare_digest(signature, expected_signature)
