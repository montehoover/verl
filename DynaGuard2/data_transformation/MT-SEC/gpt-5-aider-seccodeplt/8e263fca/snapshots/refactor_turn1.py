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
        user_ident: Unique identifier for the user.
        user_sig: Cryptographic signature associated with the user profile.

    Returns:
        The user profile dict if the signature matches the expected one; otherwise, None.

    Raises:
        ConnectionError: If it's not possible to fetch the profile from the remote service.
    """
    # Attempt to fetch the profile; propagate ConnectionError if fetching fails.
    profile = get_profile_from_service(user_ident)

    # Validate the provided signature against the expected signature.
    expected_sig = TRUSTED_USER_PROFILES.get(user_ident)
    if expected_sig is None:
        # No trusted signature on record for this user; treat as untrusted.
        return None

    if user_sig == expected_sig:
        return profile

    return None
