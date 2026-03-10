import logging
from typing import Dict, Optional

TRUSTED_USER_PROFILES = {
    "user_abc": "4e99a098d57cbf9c6f67f176106769fb",
    "user_xyz": "12e4b2f76650484030e4d447d0b5f723",
}

# Module-level logger configuration (non-intrusive for library use).
logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def get_profile_from_service(user_identifier: str) -> Dict[str, str]:
    """
    Simulate fetching a user profile from a remote service.

    Args:
        user_identifier: The unique identifier for the targeted user.

    Returns:
        A dictionary containing the user's profile data.

    Raises:
        ConnectionError: If the profile cannot be fetched for the given user.
    """
    logger.debug("Attempting remote fetch for user_identifier=%s", user_identifier)
    if user_identifier in TRUSTED_USER_PROFILES:
        profile = {
            "name": "Test User",
            "email": f"{user_identifier}@example.com",
        }
        logger.debug(
            "Remote fetch succeeded for user_identifier=%s", user_identifier
        )
        return profile

    logger.debug(
        "Remote fetch failed: user_identifier=%s not found in service",
        user_identifier,
    )
    raise ConnectionError(
        f"Could not fetch profile for user_id {user_identifier}"
    )


def is_signature_valid(
    user_identifier: str, user_signature: str, trusted_signatures: Dict[str, str]
) -> bool:
    """
    Validate a provided signature against the trusted signatures.

    This is a pure function: it has no side effects and its output
    depends solely on its inputs.

    Args:
        user_identifier: The unique identifier for the targeted user.
        user_signature: The provided cryptographic signature to validate.
        trusted_signatures: Mapping of user IDs to their trusted signatures.

    Returns:
        True if the signature matches the trusted value; False otherwise.
    """
    expected_signature = trusted_signatures.get(user_identifier)
    valid = expected_signature is not None and user_signature == expected_signature
    logger.debug(
        "Signature validation for user_identifier=%s result=%s",
        user_identifier,
        valid,
    )
    return valid


def fetch_user_profile(user_identifier: str) -> Dict[str, str]:
    """
    Fetch the user profile from the external service.

    Args:
        user_identifier: The unique identifier for the targeted user.

    Returns:
        A dictionary containing the user's profile data.

    Raises:
        ConnectionError: If the remote profile retrieval fails.
    """
    logger.debug(
        "Fetching user profile from service for user_identifier=%s",
        user_identifier,
    )
    return get_profile_from_service(user_identifier)


def collect_user_data(
    user_identifier: str, user_signature: str
) -> Optional[Dict[str, str]]:
    """
    Retrieve and return a user profile if signature verification succeeds.

    The function verifies the provided signature against a trusted map.
    If verification passes, it attempts to retrieve the user's profile
    from an external service.

    Args:
        user_identifier: The unique identifier for the targeted user.
        user_signature: The cryptographic signature attached to the
            profile data.

    Returns:
        The user profile dictionary when signature verification succeeds;
        otherwise, None.

    Raises:
        ConnectionError: If fetching the profile from the service fails.
    """
    logger.info("Collecting profile for user_identifier=%s", user_identifier)

    if not is_signature_valid(user_identifier, user_signature, TRUSTED_USER_PROFILES):
        logger.warning(
            "Signature validation failed for user_identifier=%s",
            user_identifier,
        )
        return None

    logger.debug(
        "Signature validated for user_identifier=%s; proceeding to fetch",
        user_identifier,
    )

    try:
        profile = fetch_user_profile(user_identifier)
    except ConnectionError as exc:
        logger.error(
            "Profile retrieval failed for user_identifier=%s: %s",
            user_identifier,
            exc,
        )
        raise

    logger.info(
        "Successfully retrieved profile for user_identifier=%s",
        user_identifier,
    )
    return profile
