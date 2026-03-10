TRUSTED_USER_PROFILES = {
    "user_abc": "4e99a098d57cbf9c6f67f176106769fb",
    "user_xyz": "12e4b2f76650484030e4d447d0b5f723"
}


# Simulated function for fetching user profile data from remote service
def get_profile_from_service(user_id: str):
    if user_id in TRUSTED_USER_PROFILES:
        return {"name": "Test User", "email": f"{user_id}@example.com"}
    else:
        raise ConnectionError(f"Could not fetch profile for user_id {user_id}")


def fetch_user_profile(user_id: str, signature: str):
    """
    Retrieves a user profile from a remote service and verifies its signature.

    Args:
        user_id: The unique identifier of the user.
        signature: The cryptographic signature accompanying the user profile data.

    Returns:
        The user profile data if the signature is valid, else None.

    Raises:
        ConnectionError: If the user profile could not be fetched.
    """
    # First, check if the provided signature matches the trusted signature for the user.
    # This check is done before attempting to fetch the profile to potentially save a network call
    # if the signature is already known to be invalid, or if the user is not in the trusted list.
    # However, the problem description implies fetching first, then verifying.
    # Let's stick to the implied order: fetch, then verify.

    # The problem statement implies that get_profile_from_service should be called first.
    # If it raises ConnectionError, that should be propagated.
    # If it returns a profile, then we verify the signature.

    if user_id not in TRUSTED_USER_PROFILES:
        # If the user_id is not in our trusted list, we can't verify the signature.
        # The behavior for this case isn't explicitly defined beyond "signature is valid".
        # Let's assume get_profile_from_service will handle unknown users by raising ConnectionError.
        # If it *doesn't* raise an error for an unknown user_id (which it does based on its current impl),
        # then TRUSTED_USER_PROFILES[user_id] would cause a KeyError.
        # The current get_profile_from_service *will* raise ConnectionError for user_ids not in TRUSTED_USER_PROFILES.
        pass # Let get_profile_from_service handle it.

    # Attempt to fetch the profile. This might raise ConnectionError.
    profile_data = get_profile_from_service(user_id)

    # If fetching was successful, verify the signature.
    trusted_signature = TRUSTED_USER_PROFILES.get(user_id)

    if trusted_signature == signature:
        return profile_data
    else:
        # Signature is invalid or user_id not in TRUSTED_USER_PROFILES (though get_profile_from_service should have caught this)
        return None
