"""
Utilities for verifying API message signatures.

This module exposes `verify_message` which validates a provided cryptographic
signature against a trusted value stored in `VALID_API_MESSAGES`, using a
constant-time comparison to mitigate timing side-channel attacks.
"""

import logging
from hmac import compare_digest


# Module-level logger for this file. A NullHandler is attached so importing this
# module does not configure logging for the entire application. The application
# using this module should configure handlers/levels/formatters as desired.
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


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

    Logging:
        - Logs an INFO message at the start of each verification attempt.
        - Logs an INFO message indicating success or failure for known IDs.
        - Logs a WARNING and re-raises if an unknown ``api_message_id`` is provided.

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
    # Log the start of a verification attempt. Do not log sensitive data such as
    # the signature itself.
    logger.info("verify_message: attempting verification for api_message_id='%s'", api_message_id)

    try:
        # Retrieve the expected signature for the provided message ID.
        # Using direct indexing ensures a KeyError is raised for unknown IDs,
        # as required by the specification.
        expected_signature = VALID_API_MESSAGES[api_message_id]
    except KeyError:
        # Log and re-raise to preserve the required behavior.
        logger.warning(
            "verify_message: unknown api_message_id='%s' (raising KeyError)",
            api_message_id,
        )
        raise

    # Perform a constant-time comparison to avoid timing attacks.
    is_valid = compare_digest(expected_signature, crypto_signature)

    # Log the outcome without exposing signature values.
    if is_valid:
        logger.info(
            "verify_message: verification successful for api_message_id='%s'",
            api_message_id,
        )
    else:
        logger.info(
            "verify_message: verification failed for api_message_id='%s'",
            api_message_id,
        )

    return is_valid
