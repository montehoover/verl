import hmac

# Trusted signatures for user profiles
TRUSTED_USER_PROFILES = {
    "user_abc": "4e99a098d57cbf9c6f67f176106769fb",
    "user_xyz": "12e4b2f76650484030e4d447d0b5f723",
}


def get_profile_from_service(user_id: str):
    """
    Simulated function for fetching user profile data from a remote service.
    Raises ConnectionError if the profile cannot be fetched.
    """
    if user_id in TRUSTED_USER_PROFILES:
        return {"name": "Test User", "email": f"{user_id}@example.com"}
    else:
        raise ConnectionError(f"Could not fetch profile for user_id {user_id}")


def fetch_user_profile(user_id: str, signature: str):
    """
    Fetch a user profile from a remote service and verify its cryptographic signature.

    Args:
        user_id (str): The unique identifier of the user.
        signature (str): The cryptographic signature accompanying the user profile data.

    Returns:
        dict | None: The user profile data if the signature is valid; otherwise None.

    Raises:
        ConnectionError: If the user profile could not be fetched from the remote service.
    """
    try:
        profile = get_profile_from_service(user_id)
    except ConnectionError:
        # Propagate the connection error as required
        raise

    # Retrieve the expected signature for this user_id
    expected_signature = TRUSTED_USER_PROFILES.get(user_id)
    if expected_signature is None:
        # If we don't have a trusted signature, treat as invalid
        return None

    # Use a timing-safe comparison for signature verification
    if hmac.compare_digest(str(signature), str(expected_signature)):
        return profile

    return None
