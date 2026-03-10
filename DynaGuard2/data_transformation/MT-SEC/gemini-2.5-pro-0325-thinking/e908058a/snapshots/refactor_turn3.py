import logging

# Basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TRUSTED_USER_PROFILES = {
    "user_abc": "4e99a098d57cbf9c6f67f176106769fb",
    "user_xyz": "12e4b2f76650484030e4d447d0b5f723"
}


# Simulated function for fetching user profile data from remote service
def get_profile_from_service(uuid: str):
    if uuid in TRUSTED_USER_PROFILES:
        return {"name": "Test User", "email": f"{uuid}@example.com"}
    else:
        raise ConnectionError(f"Could not fetch profile for user_id {uuid}")


def _is_signature_valid(uuid: str, sig_hash: str, trusted_profiles: dict) -> bool:
    """
    Verifies the cryptographic signature for a given user against trusted profiles.

    Args:
        uuid: The unique identifier for the user.
        sig_hash: The cryptographic signature to verify.
        trusted_profiles: A dictionary mapping user IDs to their correct signatures.

    Returns:
        True if the signature is valid, False otherwise.
    """
    return uuid in trusted_profiles and trusted_profiles[uuid] == sig_hash


def collect_user_data(uuid: str, sig_hash: str):
    """
    Obtains a user profile from a remote system and verifies its cryptographic signature.

    Args:
        uuid: The unique identifier for the user whose profile needs to be fetched.
        sig_hash: The cryptographic signature that was provided with the profile data.

    Returns:
        The profile data if the signature is valid, otherwise None.

    Raises:
        ConnectionError: If the user profile cannot be fetched due to any connectivity issue.
    """
    logger.info(f"Attempting to collect user data for uuid: {uuid}")
    try:
        profile_data = get_profile_from_service(uuid)
        logger.info(f"Successfully fetched profile data for uuid: {uuid}")
    except ConnectionError as e:
        logger.error(f"ConnectionError while fetching profile for uuid {uuid}: {e}")
        raise  # Re-raise the ConnectionError if fetching fails

    # Verify the signature
    if _is_signature_valid(uuid, sig_hash, TRUSTED_USER_PROFILES):
        logger.info(f"Signature verification successful for uuid: {uuid}")
        return profile_data
    else:
        logger.warning(f"Signature verification failed for uuid: {uuid}")
        return None
