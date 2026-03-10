import logging

# Basic logging configuration for demonstration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TRUSTED_USER_PROFILES = {
    "user_abc": "4e99a098d57cbf9c6f67f176106769fb",
    "user_xyz": "12e4b2f76650484030e4d447d0b5f723"
}


# Simulated function for fetching user profile data from remote service
def get_profile_from_service(uid: str):
    if uid in TRUSTED_USER_PROFILES:
        return {"name": "Test Individual", "email": f"{uid}@example.com"}
    else:
        raise ConnectionError(f"Could not fetch profile for uid {uid}")


def _is_signature_valid(uid: str, sig: str, trusted_map: dict) -> bool:
    """
    Validates the cryptographic signature for a given user ID against a trusted map.
    This is a pure function, relying only on its inputs.

    Args:
        uid: The unique identifier for the user.
        sig: The cryptographic signature to validate.
        trusted_map: A dictionary mapping user IDs to their correct signatures.

    Returns:
        True if the signature is valid, False otherwise.
    """
    expected_sig = trusted_map.get(uid)
    return expected_sig is not None and expected_sig == sig


def retrieve_user_profile(uid: str, sig: str):
    """
    Retrieves a user profile from a remote service and validates its signature.

    Args:
        uid: The unique identifier for the user.
        sig: The cryptographic signature associated with the user profile.

    Returns:
        The user profile if the signature is valid, otherwise None.

    Raises:
        ConnectionError: If the profile cannot be fetched from the remote service.
    """
    logger.info(f"Attempting to retrieve profile for user ID: {uid}")
    try:
        # Step 1: Fetch profile data from the service.
        # get_profile_from_service is treated as the separate "profile fetching logic".
        profile_data = get_profile_from_service(uid)
        logger.info(f"Successfully fetched raw profile data for user ID: {uid}")
    except ConnectionError as e:
        logger.error(f"ConnectionError while fetching profile for user ID {uid}: {e}")
        raise  # Re-raise the exception after logging

    # Step 2: Validate the signature using the pure helper function.
    signature_is_valid = _is_signature_valid(uid, sig, TRUSTED_USER_PROFILES)

    if signature_is_valid:
        logger.info(f"Signature validated successfully for user ID: {uid}")
        logger.info(f"Profile retrieval successful for user ID: {uid}")
        return profile_data
    else:
        logger.warning(f"Signature validation failed for user ID: {uid}")
        logger.info(f"Profile retrieval failed for user ID: {uid} due to invalid signature.")
        # Signature is invalid (either uid not in trusted_map for sig check, or sig mismatch)
        return None
