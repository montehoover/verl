"""
Utilities for verifying the authenticity of API messages by validating
their cryptographic signatures against a trusted, in-memory registry.

This module includes logging to record each verification attempt, including
the message ID and whether verification succeeded or failed.
"""

import logging

# Module-level logger. The application using this module should configure
# logging (handlers/formatters/levels). We avoid calling basicConfig here
# to prevent unexpected global side effects.
logger = logging.getLogger(__name__)

# A trusted registry mapping message IDs to their expected cryptographic
# signatures. In production, this might be backed by a database or secure
# configuration store.
VALID_API_MESSAGES = {
    "msg_1001": "abcd1234efgh5678ijkl9012mnop3456",
    "msg_1002": "1122aabbccdd3344eeff5566gggghhhh"
}


def verify_api_message(msg_id: str, sign: str) -> bool:
    """
    Verify the authenticity of a message from an external API.

    The function looks up the expected cryptographic signature for the given
    message ID and compares it with the provided signature. Each verification
    attempt is logged with the message ID and the outcome.

    Args:
        msg_id (str): The unique identifier of the message to verify.
        sign (str): The cryptographic signature provided with the message.

    Returns:
        bool: True if the signature matches the stored signature for msg_id;
        otherwise, False.

    Raises:
        KeyError: If msg_id is not present in VALID_API_MESSAGES.
    """
    # Log the start of a verification attempt for visibility and diagnostics.
    logger.debug("Starting verification attempt for msg_id=%s", msg_id)

    # Ensure the message ID exists in the trusted registry; if not, raise an
    # explicit KeyError to indicate the message ID is unknown.
    try:
        expected_signature = VALID_API_MESSAGES[msg_id]
    except KeyError as exc:
        logger.warning("Verification failed: unknown msg_id=%s", msg_id)
        raise KeyError(f"Message ID not found: {msg_id}") from exc

    # Compare the provided signature to the expected one.
    is_valid = expected_signature == sign

    # Log the verification result. Avoid logging sensitive signature values.
    if is_valid:
        logger.info("Verification succeeded for msg_id=%s", msg_id)
    else:
        logger.info("Verification failed for msg_id=%s", msg_id)

    return is_valid
