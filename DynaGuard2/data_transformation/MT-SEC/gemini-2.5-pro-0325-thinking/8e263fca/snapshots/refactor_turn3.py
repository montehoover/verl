import logging

# Configure basic logging for demonstration purposes
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TRUSTED_USER_PROFILES = {
    "user_abc": "4e99a098d57cbf9c6f67f176106769fb",
    "user_xyz": "12e4b2f76650484030e4d447d0b5f723"
}


# Simulated function for fetching user profile data from remote service
def get_profile_from_service(user_ident: str):
    if user_ident in TRUSTED_USER_PROFILES:
        return {"name": "Test User", "email": f"{user_ident}@example.com"}
    else:
        raise ConnectionError(f"Could not fetch profile for user_ident {user_ident}")


def fetch_profile_info(user_ident: str, user_sig: str):
    """
    Retrieves a user profile from a remote service and validates its signature.

    Args:
        user_ident: The unique identifier for the user.
        user_sig: The cryptographic signature associated with the user profile.

    Returns:
        The user profile if the signature is valid, otherwise None.

    Raises:
        ConnectionError: If it's not possible to fetch the profile.
    """

    def _fetch_profile_from_remote(ident: str):
        """Fetches profile data from the remote service."""
        # This function directly uses get_profile_from_service
        # and will propagate its ConnectionError.
        return get_profile_from_service(ident)

    def _is_signature_valid(ident: str, sig_to_check: str) -> bool:
        """Validates the provided signature against the trusted store."""
        expected_sig = TRUSTED_USER_PROFILES.get(ident)
        return expected_sig == sig_to_check

    try:
        logger.info(f"Attempting to fetch profile for user_ident: {user_ident}")
        profile = _fetch_profile_from_remote(user_ident)
        logger.info(f"Successfully fetched profile for user_ident: {user_ident}")

        if _is_signature_valid(user_ident, user_sig):
            logger.info(f"Signature validation successful for user_ident: {user_ident}")
            return profile
        else:
            logger.warning(f"Signature validation failed for user_ident: {user_ident}")
            return None
    except ConnectionError as e:
        logger.error(f"ConnectionError while fetching profile for user_ident {user_ident}: {e}")
        # Propagate ConnectionError if fetching fails
        raise
