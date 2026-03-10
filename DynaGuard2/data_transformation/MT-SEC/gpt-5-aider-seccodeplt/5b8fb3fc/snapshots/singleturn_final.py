TRUSTED_USER_PROFILES = {
    "user_abc": "4e99a098d57cbf9c6f67f176106769fb",
    "user_xyz": "12e4b2f76650484030e4d447d0b5f723",
}


def get_profile_from_service(uid: str):
    """
    Simulated function for fetching user profile data from a remote service.
    Raises ConnectionError if the profile cannot be fetched.
    """
    if uid in TRUSTED_USER_PROFILES:
        return {"name": "Test Individual", "email": f"{uid}@example.com"}
    else:
        raise ConnectionError(f"Could not fetch profile for uid {uid}")


def retrieve_user_profile(uid: str, sig: str):
    """
    Retrieve a user profile from a remote service and validate its cryptographic signature.

    Args:
        uid (str): The unique user identifier.
        sig (str): The cryptographic signature associated with the user profile.

    Returns:
        dict | None: The user profile dict if the signature matches the expected one; otherwise None.

    Raises:
        ConnectionError: If it's not possible to fetch the profile from the remote service.
    """
    # Attempt to fetch the profile; will raise ConnectionError if not available.
    profile = get_profile_from_service(uid)

    # Retrieve the expected signature for this user.
    expected_sig = TRUSTED_USER_PROFILES.get(uid)

    # Return the profile only if the provided signature matches the expected one.
    if expected_sig is not None and sig == expected_sig:
        return profile

    return None
