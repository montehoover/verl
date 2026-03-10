from typing import Optional, Dict, Any, Mapping, Callable
import logging

logger = logging.getLogger(__name__)

TRUSTED_USER_PROFILES: Dict[str, str] = {
    "user_abc": "4e99a098d57cbf9c6f67f176106769fb",
    "user_xyz": "12e4b2f76650484030e4d447d0b5f723"
}


# Simulated function for fetching user profile data from remote service
def get_profile_from_service(uuid: str) -> Dict[str, Any]:
    if uuid in TRUSTED_USER_PROFILES:
        return {"name": "Test User", "email": f"{uuid}@example.com"}
    else:
        raise ConnectionError(f"Could not fetch profile for user_id {uuid}")


def is_signature_valid(uuid: str, sig_hash: str, trusted_map: Mapping[str, str] = TRUSTED_USER_PROFILES) -> bool:
    """
    Check whether the provided signature matches the trusted signature for the user.

    Args:
        uuid: The unique identifier for the user.
        sig_hash: The cryptographic signature to verify.
        trusted_map: Mapping of user IDs to their trusted signatures.

    Returns:
        True if the signature is valid; otherwise, False.
    """
    expected_sig = trusted_map.get(uuid)
    return expected_sig is not None and sig_hash == expected_sig


def fetch_user_profile(
    uuid: str,
    fetcher: Callable[[str], Dict[str, Any]] = get_profile_from_service
) -> Dict[str, Any]:
    """
    Fetch the user profile using the provided fetcher function.

    Args:
        uuid: The unique identifier for the user.
        fetcher: A callable that, given a uuid, returns the profile dict.
                 May raise ConnectionError on connectivity issues.

    Returns:
        The fetched profile dictionary.

    Raises:
        ConnectionError: If the profile cannot be fetched due to connectivity issues.
    """
    return fetcher(uuid)


def collect_user_data(uuid: str, sig_hash: str) -> Optional[Dict[str, Any]]:
    """
    Obtain a user profile from a remote system and verify its cryptographic signature.

    Args:
        uuid: The unique identifier for the user whose profile needs to be fetched.
        sig_hash: The cryptographic signature that was provided with the profile data.

    Returns:
        The profile data if the signature is valid; otherwise, None.

    Raises:
        ConnectionError: If the user profile cannot be fetched due to any connectivity issue.
    """
    logger.info("Starting user data collection for user_id='%s'", uuid)

    valid = is_signature_valid(uuid, sig_hash)
    if not valid:
        logger.warning("Signature verification failed for user_id='%s'", uuid)
        return None

    logger.debug("Signature verification succeeded for user_id='%s'", uuid)

    try:
        logger.debug("Attempting to fetch profile for user_id='%s'", uuid)
        profile = fetch_user_profile(uuid)
        logger.info("Successfully fetched profile for user_id='%s'", uuid)
        return profile
    except ConnectionError as exc:
        logger.error("Error fetching profile for user_id='%s': %s", uuid, exc)
        raise
