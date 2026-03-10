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
        ConnectionError: If it's not possible to fetch the profile from the remote service.
    """
    expected_sig = TRUSTED_USER_PROFILES.get(user_ident)

    if expected_sig is None or user_sig != expected_sig:
        return None

    # If the signature is valid, try to fetch the profile.
    # get_profile_from_service will raise ConnectionError if it fails.
    profile_data = get_profile_from_service(user_ident)
    return profile_data
