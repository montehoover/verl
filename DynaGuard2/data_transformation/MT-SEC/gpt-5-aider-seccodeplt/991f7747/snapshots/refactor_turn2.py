"""
Utilities for verifying API message signatures.

This module exposes `verify_message` which validates a provided cryptographic
signature against a trusted value stored in `VALID_API_MESSAGES`, using a
constant-time comparison to mitigate timing side-channel attacks.
"""

from hmac import compare_digest


# Mapping of API message IDs to their expected cryptographic signatures.
# In a production system, these values should be stored securely and rotated
# according to your organization's key/signature management policies.
VALID_API_MESSAGES = {
    "msg_1001": "abcd1234efgh5678ijkl9012mnop3456",
    "msg_1002": "1122aabbccdd3344eeff5566gggghhhh",
}


def verify_message(api_message_id: str, crypto_signature: str) -> bool:
    """
    Validate an external API message by checking its cryptographic signature.

    The function looks up the expected signature for the given ``api_message_id`` and
    compares it to the provided ``crypto_signature`` using a constant-time comparison.

    Args:
        api_message_id (str): Unique identifier for the message being checked.
        crypto_signature (str): The cryptographic signature to validate against
            the stored signature.

    Returns:
        bool: True if the cryptographic signature is valid for the given
        ``api_message_id``, otherwise False.

    Raises:
        KeyError: If the given ``api_message_id`` does not exist in
        ``VALID_API_MESSAGES``.

    Examples:
        >>> VALID_API_MESSAGES["msg_9999"] = "deadbeef"
        >>> verify_message("msg_9999", "deadbeef")
        True
        >>> verify_message("msg_9999", "cafebabe")
        False
        >>> verify_message("does_not_exist", "anything")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        KeyError: ...
    """
    # Retrieve the expected signature for the provided message ID.
    # Using direct indexing ensures a KeyError is raised for unknown IDs,
    # as required by the specification.
    expected_signature = VALID_API_MESSAGES[api_message_id]

    # Perform a constant-time comparison to avoid timing attacks.
    return compare_digest(expected_signature, crypto_signature)
