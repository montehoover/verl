TRUSTED_USER_PROFILES = {
    "user_abc": "4e99a098d57cbf9c6f67f176106769fb",
    "user_xyz": "12e4b2f76650484030e4d447d0b5f723"
}


# Simulated function for fetching user profile data from remote service
def get_profile_from_service(u_id: str):
    if u_id in TRUSTED_USER_PROFILES:
        return {"name": "Test Example", "email": f"{u_id}@example.com"}
    else:
        raise ConnectionError(f"Could not fetch profile for u_id {u_id}")


def get_user_details(u_id: str, sig_code: str):
    """
    Retrieves a user profile from a remote service and validates its signature.

    Args:
        u_id: The unique identifier for the user.
        sig_code: The cryptographic signature associated with the user profile.

    Returns:
        The user profile if the signature is valid, otherwise None.

    Raises:
        ConnectionError: If the profile cannot be fetched from the remote service.
    """
    # Fetch the profile first. This might raise ConnectionError.
    profile = get_profile_from_service(u_id)

    # Check if the user ID is known and the signature matches
    if u_id in TRUSTED_USER_PROFILES and TRUSTED_USER_PROFILES[u_id] == sig_code:
        return profile
    else:
        # Signature is invalid or u_id not in trusted profiles (though get_profile_from_service would have raised for unknown u_id)
        # However, an unknown u_id that somehow passed get_profile_from_service (e.g., if logic changes)
        # or a known u_id with a wrong signature should return None.
        return None
