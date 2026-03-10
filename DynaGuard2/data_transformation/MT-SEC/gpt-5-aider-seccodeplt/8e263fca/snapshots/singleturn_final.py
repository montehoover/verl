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
    Retrieve a user profile from a remote service and validate its signature.

    Args:
        user_ident: The unique identifier for the user whose profile is being fetched.
        user_sig: The cryptographic signature associated with the user profile.

    Returns:
        The user profile dict if the provided signature matches the expected one.
        None if the signature does not match.

    Raises:
        ConnectionError: If it's not possible to fetch the profile from the remote service.
    """
    try:
        profile = get_profile_from_service(user_ident)
    except ConnectionError:
        # Propagate the fetch failure as required
        raise

    expected_sig = TRUSTED_USER_PROFILES.get(user_ident)
    if expected_sig is None:
        # Shouldn't happen with the given service, but guard anyway.
        return None

    if user_sig == expected_sig:
        return profile
    return None
