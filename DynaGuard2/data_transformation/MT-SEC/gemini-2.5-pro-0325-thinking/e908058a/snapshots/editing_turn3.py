TRUSTED_USER_PROFILES = {
    "user_abc": "4e99a098d57cbf9c6f67f176106769fb",
    "user_xyz": "12e4b2f76650484030e4d447d0b5f723"
}

def get_profile_from_service(uuid: str):
    if uuid in TRUSTED_USER_PROFILES:
        return {"name": "Test User", "email": f"{uuid}@example.com"}
    else:
        raise ConnectionError(f"Could not fetch profile for user_id {uuid}")

def collect_user_data(uuid: str, sig_hash: str):
    """
    Fetches a user profile from a remote system and verifies it using a
    cryptographic signature.

    Args:
        uuid: The unique identifier for the user.
        sig_hash: The cryptographic signature.

    Returns:
        The profile data if the signature is valid, otherwise None.

    Raises:
        ConnectionError: If there's a connectivity issue.
    """
    expected_sig_hash = TRUSTED_USER_PROFILES.get(uuid)
    if expected_sig_hash is None or sig_hash != expected_sig_hash:
        return None

    # If we reach here, the signature is valid for a known user,
    # or the user is not in TRUSTED_USER_PROFILES (in which case expected_sig_hash was None).
    # The problem statement implies we should only proceed if the uuid is in TRUSTED_USER_PROFILES
    # and the sig_hash matches. The check above handles this.

    # Now, attempt to fetch the profile.
    # get_profile_from_service will raise ConnectionError if uuid is not in its "service"
    # which is also TRUSTED_USER_PROFILES in this example.
    # If uuid was not in TRUSTED_USER_PROFILES, the first check `expected_sig_hash is None`
    # would have caused a return None already.
    # So, if we are here, uuid must be in TRUSTED_USER_PROFILES.
    profile_data = get_profile_from_service(uuid)
    return profile_data
