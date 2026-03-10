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


def retrieve_user_profile(uid: str, sig: str):
    """
    Retrieves a user profile from a remote service and validates its signature.

    Args:
        uid: The unique identifier for the user.
        sig: The cryptographic signature associated with the user profile.

    Returns:
        The user profile if the signature is valid, otherwise None.

    Raises:
        ConnectionError: If it's not possible to fetch the profile.
    """
    # Fetch the profile first. This might raise ConnectionError.
    profile = get_profile_from_service(uid)

    # Check if the uid exists in trusted profiles and if the signature matches
    if uid in TRUSTED_USER_PROFILES and TRUSTED_USER_PROFILES[uid] == sig:
        return profile
    else:
        # Signature is invalid or uid is not in trusted list (though get_profile_from_service would have failed first for unknown uid)
        return None
