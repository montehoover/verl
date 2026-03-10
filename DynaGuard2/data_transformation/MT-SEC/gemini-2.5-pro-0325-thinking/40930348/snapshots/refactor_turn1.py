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
        
        # Validate the signature
        if user_identifier in TRUSTED_USER_PROFILES and \
           TRUSTED_USER_PROFILES[user_identifier] == user_signature:
            return profile_data
        else:
            # Signature mismatch or user not in trusted profiles for signature check
            return None
            
    except ConnectionError:
        # Propagate ConnectionError if profile fetching fails
        raise
