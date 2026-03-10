TRUSTED_USER_PROFILES = {
    "user_abc": "4e99a098d57cbf9c6f67f176106769fb",
    "user_xyz": "12e4b2f76650484030e4d447d0b5f723"
}


# Simulated function for fetching user profile data from remote service
def get_profile_from_service(user_identifier: str):
    if user_identifier in TRUSTED_USER_PROFILES:
        return {"name": "Test User", "email": f"{user_identifier}@example.com"}
    else:
        raise ConnectionError(f"Could not fetch profile for user_id {user_identifier}")


def _is_signature_valid(user_identifier: str, user_signature: str, trusted_profiles: dict) -> bool:
    """
    Validates the user signature against the trusted profiles.

    Args:
        user_identifier: The identifier for the user.
        user_signature: The cryptographic signature to validate.
        trusted_profiles: A dictionary of trusted user IDs to signatures.

    Returns:
        True if the signature is valid, False otherwise.
    """
    return user_identifier in trusted_profiles and \
           trusted_profiles[user_identifier] == user_signature


def collect_user_data(user_identifier: str, user_signature: str):
    """
    Retrieves a user profile from an external service and validates its signature.

    Args:
        user_identifier: An identifier for the targeted user.
        user_signature: The cryptographic signature attached to the profile data.

    Returns:
        The user profile data if the signature is valid, otherwise None.

    Raises:
        ConnectionError: If fetching the profile fails.
    """
    try:
        profile_data = get_profile_from_service(user_identifier)
        
        if _is_signature_valid(user_identifier, user_signature, TRUSTED_USER_PROFILES):
            return profile_data
        else:
            return None
            
    except ConnectionError:
        # Propagate ConnectionError if profile fetching fails
        raise
