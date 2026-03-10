"""
Utilities to validate API messages using cryptographic signatures.

This module exposes validate_api_message(), which checks that a provided
cryptographic signature matches the expected value for a given message ID,
using a constant-time comparison to avoid timing leaks.
"""

from hmac import compare_digest


# Known-good signatures keyed by external API message IDs.
# NOTE: In production, store these secrets in a secure data store rather than
# in source code or environment variables checked into version control.
VALID_API_MESSAGES = {
    "msg_1001": "abcd1234efgh5678ijkl9012mnop3456",
    "msg_1002": "1122aabbccdd3344eeff5566gggghhhh",
}


def validate_api_message(api_message_id: str, cryptographic_sign: str) -> bool:
    """
    Validate an API message's authenticity via its cryptographic signature.

    The function looks up the expected signature for the provided message ID in
    the global VALID_API_MESSAGES mapping and compares it to the supplied
    signature using a constant-time equality check.

    Args:
        api_message_id: The message ID whose authenticity is being verified.
        cryptographic_sign: The signature purportedly associated with the
            message ID.

    Returns:
        True if the provided signature matches the expected signature for the
        given message ID; otherwise, False.

    Raises:
        KeyError: If the message ID does not exist in VALID_API_MESSAGES.

    Examples:
        >>> validate_api_message("msg_1001", "abcd1234efgh5678ijkl9012mnop3456")
        True
        >>> validate_api_message("msg_1001", "wrong")
        False
        >>> validate_api_message("unknown", "anything")
        Traceback (most recent call last):
        ...
        KeyError: 'unknown'
    """
    # Retrieve the expected signature. Let KeyError propagate if the ID is not
    # present, as missing IDs are considered exceptional.
    expected_signature = VALID_API_MESSAGES[api_message_id]

    # Compare in constant time to prevent timing side-channel leakage.
    return compare_digest(cryptographic_sign, expected_signature)
