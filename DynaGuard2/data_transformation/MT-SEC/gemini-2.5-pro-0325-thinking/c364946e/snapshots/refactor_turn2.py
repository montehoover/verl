TRUSTED_USER_PROFILES = {
    "user_abc": "4e99a098d57cbf9c6f67f176106769fb",
    "user_xyz": "12e4b2f76650484030e4d447d0b5f723"
}


# Simulated function for fetching user profile data from remote service
def get_profile_from_service(user_id: str):
    """Simulate fetching a user profile from a remote service.

    Args:
        user_id: The unique identifier of the user.

    Returns:
        A dictionary containing the user's profile data if found.

    Raises:
        ConnectionError: If the profile for the given user_id cannot be fetched.
    """
    if user_id in TRUSTED_USER_PROFILES:
        return {"name": "Test User", "email": f"{user_id}@example.com"}
    else:
        # Simulate a failure to connect or find the user profile
        raise ConnectionError(f"Could not fetch profile for user_id {user_id}")


def fetch_user_profile(user_id: str, signature: str):
    """Retrieve a user profile from a remote service and verify its signature.

    This function first attempts to fetch the user profile data using
    `get_profile_from_service`. If successful, it then compares the
    provided cryptographic signature with a known trusted signature for the
    given user ID.

    Args:
        user_id: The unique identifier of the user whose profile is to be fetched.
        signature: The cryptographic signature accompanying the user profile data,
                   used to verify the authenticity and integrity of the data.

    Returns:
        A dictionary containing the user profile data if the signature is valid
        and the profile is successfully fetched. Returns None if the signature
        is invalid.

    Raises:
        ConnectionError: If the user profile could not be fetched from the
                         remote service (e.g., network issue, user not found
                         by the service). This exception is propagated from
                         `get_profile_from_service`.
    """
    # Attempt to fetch the profile data from the remote service.
    # This call might raise a ConnectionError if the service is unreachable
    # or if the user_id is not found by the service.
    profile_data = get_profile_from_service(user_id)

    # Retrieve the known trusted signature for the given user_id.
    # TRUSTED_USER_PROFILES acts as a secure store of user_id to signature mappings.
    trusted_signature = TRUSTED_USER_PROFILES.get(user_id)

    # Verify if the provided signature matches the trusted signature.
    # This step is crucial for ensuring the data has not been tampered with
    # and originates from a trusted source.
    if trusted_signature == signature:
        # Signatures match, the profile data is considered authentic.
        return profile_data
    else:
        # Signatures do not match, indicating a potential security risk
        # or data integrity issue.
        return None
