import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TRUSTED_USER_PROFILES = {
    "user_abc": "4e99a098d57cbf9c6f67f176106769fb",
    "user_xyz": "12e4b2f76650484030e4d447d0b5f723"
}


# Simulated function for fetching user profile data from remote service
def get_profile_from_service(u_id: str):
    if u_id in TRUSTED_USER_PROFILES:
        return {"name": "Test Example", "email": f"{u_id}@example.com"}
    else:
        raise ConnectionError(f"Could not fetch profile for u_id {u_id}")


def _is_signature_valid(u_id: str, sig_code: str, trusted_profiles: dict) -> bool:
    """
    Validates the provided signature for a given user ID against trusted profiles.

    Args:
        u_id: The unique identifier for the user.
        sig_code: The cryptographic signature to validate.
        trusted_profiles: A dictionary mapping user IDs to their correct signatures.

    Returns:
        True if the signature is valid, False otherwise.
    """
    return u_id in trusted_profiles and trusted_profiles[u_id] == sig_code


def get_user_details(u_id: str, sig_code: str):
    """
    Retrieves a user profile from a remote service and validates its signature.

    Args:
        u_id: The unique identifier for the user.
        sig_code: The cryptographic signature associated with the user profile.

    Returns:
        The user profile if the signature is valid, otherwise None.

    Raises:
        ConnectionError: If the profile cannot be fetched from the remote service.
    """
    logger.info(f"Attempting to get user details for u_id: {u_id}")
    try:
        # Fetch the profile first. This might raise ConnectionError.
        profile = get_profile_from_service(u_id)
        logger.info(f"Successfully fetched profile for u_id: {u_id}")
    except ConnectionError as e:
        logger.error(f"ConnectionError for u_id {u_id}: {e}")
        raise  # Re-raise the exception after logging

    # Validate the signature
    if _is_signature_valid(u_id, sig_code, TRUSTED_USER_PROFILES):
        logger.info(f"Signature validation successful for u_id: {u_id}")
        return profile
    else:
        logger.warning(f"Signature validation failed for u_id: {u_id}")
        # Signature is invalid
        return None
