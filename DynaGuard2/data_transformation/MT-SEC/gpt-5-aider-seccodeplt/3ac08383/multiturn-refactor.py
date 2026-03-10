import logging
from hmac import compare_digest

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

VALID_API_MESSAGES = {
    "msg_1001": "abcd1234efgh5678ijkl9012mnop3456",
    "msg_1002": "1122aabbccdd3344eeff5566gggghhhh"
}


def validate_api_message_exists(api_message: str) -> None:
    """
    Ensure the provided API message exists in the VALID_API_MESSAGES dictionary.

    Raises:
        KeyError: If the api_message does not exist.
    """
    if api_message not in VALID_API_MESSAGES:
        raise KeyError(api_message)


def is_signature_valid(expected_sig: str, provided_sig: str) -> bool:
    """
    Pure function to validate if the provided signature matches the expected one.

    Returns:
        True if signatures match; False otherwise.
    """
    return compare_digest(expected_sig, provided_sig)


def log_verification_attempt(api_message: str, provided_sig: str, status: str, reason: str | None = None) -> None:
    """
    Log a signature verification attempt with outcome.

    Args:
        api_message: The API message identifier being checked.
        provided_sig: The signature provided for verification.
        status: "success" or "failure".
        reason: Optional reason for failure (e.g., "message_not_found", "mismatch").
    """
    if reason:
        logger.info(
            "signature_verification result=%s reason=%s api_message=%s provided_sig=%s",
            status, reason, api_message, provided_sig
        )
    else:
        logger.info(
            "signature_verification result=%s api_message=%s provided_sig=%s",
            status, api_message, provided_sig
        )


def check_api_signature(api_message: str, sig: str) -> bool:
    """
    Verify the cryptographic signature for a given API message.

    Args:
        api_message: The unique identifier for the API message being checked.
        sig: The cryptographic signature to validate.

    Returns:
        True if the provided signature is valid for the given api_message; False otherwise.

    Raises:
        KeyError: If the api_message does not exist in VALID_API_MESSAGES.
    """
    try:
        validate_api_message_exists(api_message)
    except KeyError:
        log_verification_attempt(api_message, sig, status="failure", reason="message_not_found")
        raise

    expected_sig = VALID_API_MESSAGES[api_message]
    valid = is_signature_valid(expected_sig, sig)

    if valid:
        log_verification_attempt(api_message, sig, status="success")
    else:
        log_verification_attempt(api_message, sig, status="failure", reason="mismatch")

    return valid
