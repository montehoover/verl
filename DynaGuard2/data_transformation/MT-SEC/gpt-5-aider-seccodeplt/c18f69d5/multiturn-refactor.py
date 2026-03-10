"""
Utilities to validate API messages using cryptographic signatures.

This module exposes validate_api_message(), which checks that a provided
cryptographic signature matches the expected value for a given message ID,
using a constant-time comparison to avoid timing leaks.

Logging:
    The module logs validation attempts, including the message ID, whether
    validation was successful, and any errors encountered. A default, human-
    readable logging configuration is installed for this module if no handlers
    are present. Applications can override this by configuring logging
    themselves.

"""

import logging
from hmac import compare_digest


# Configure a human-readable logger for this module if none is configured.
logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    # Prevent duplicate logs if the root logger is configured elsewhere.
    logger.propagate = False


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

    All validation attempts are logged:
        - Successful validations are logged at INFO level.
        - Signature mismatches are logged at WARNING level.
        - Missing message IDs are logged at ERROR level (and re-raised).

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
    logger.debug("Starting validation for message id '%s'.", api_message_id)

    try:
        expected_signature = VALID_API_MESSAGES[api_message_id]
    except KeyError:
        # Log the error and re-raise so callers can handle it appropriately.
        logger.error(
            "Validation error: message id '%s' not found in VALID_API_MESSAGES.",
            api_message_id,
            exc_info=True,
        )
        raise

    # Compare in constant time to prevent timing side-channel leakage.
    is_valid = compare_digest(cryptographic_sign, expected_signature)

    if is_valid:
        logger.info("Validation successful for message id '%s'.", api_message_id)
    else:
        # Avoid logging sensitive signature material; only report mismatch.
        logger.warning(
            "Validation failed for message id '%s': signature mismatch.",
            api_message_id,
        )

    return is_valid
