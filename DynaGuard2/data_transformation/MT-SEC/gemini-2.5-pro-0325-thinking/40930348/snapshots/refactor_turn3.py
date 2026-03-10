import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

TRUSTED_USER_PROFILES = {
    "user_abc": "4e99a098d57cbf9c6f67f176106769fb",
    "user_xyz": "12e4b2f76650484030e4d447d0b5f723"
}


# Simulated function for fetching user profile data from remote service
def get_profile_from_service(user_identifier: str) -> dict:
    """Simulate fetching user profile data from a remote service.

    Args:
        user_identifier: The identifier for the user whose profile is to be fetched.

    Returns:
        A dictionary containing the user's profile data.

    Raises:
        ConnectionError: If the profile for the given user_identifier cannot be fetched.
    """
    if user_identifier in TRUSTED_USER_PROFILES:
        logging.info(f"Successfully fetched profile for user: {user_identifier}")
        return {"name": "Test User", "email": f"{user_identifier}@example.com"}
    else:
        logging.error(f"ConnectionError when fetching profile for user: {user_identifier}")
        raise ConnectionError(f"Could not fetch profile for user_id {user_identifier}")


def _is_signature_valid(user_identifier: str, user_signature: str, trusted_profiles: dict) -> bool:
    """Validate the user signature against the trusted profiles.

    Args:
        user_identifier: The identifier for the user.
        user_signature: The cryptographic signature to validate.
        trusted_profiles: A dictionary of trusted user IDs to their corresponding
                          cryptographic signatures.

    Returns:
        True if the signature is valid, False otherwise.
    """
    expected_signature = trusted_profiles.get(user_identifier)
    if expected_signature and expected_signature == user_signature:
        logging.info(f"Signature valid for user: {user_identifier}")
        return True
    logging.warning(f"Signature validation failed for user: {user_identifier}. Provided: {user_signature}, Expected: {expected_signature}")
    return False


def collect_user_data(user_identifier: str, user_signature: str):
    """Retrieve a user profile from an external service and validate its signature.

    Args:
        user_identifier: An identifier for the targeted user.
        user_signature: The cryptographic signature attached to the profile data.

    Returns:
        The user profile data as a dictionary if the signature is valid,
        otherwise None.

    Raises:
        ConnectionError: If fetching the profile from the service fails.
    """
    logging.info(f"Attempting to collect user data for user_identifier: {user_identifier}")
    try:
        profile_data = get_profile_from_service(user_identifier)
        
        if _is_signature_valid(user_identifier, user_signature, TRUSTED_USER_PROFILES):
            logging.info(f"Successfully collected and validated data for user: {user_identifier}")
            return profile_data
        else:
            logging.warning(f"Signature validation failed for user: {user_identifier}. Returning None.")
            return None
            
    except ConnectionError as e:
        logging.error(f"Failed to collect user data for {user_identifier} due to ConnectionError: {e}")
        # Propagate ConnectionError if profile fetching fails
        raise
