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
    Retrieves a user profile from a remote service and verifies the signature.
    
    Args:
        user_id: str, the unique identifier of the user whose profile needs to be fetched
        signature: str, the cryptographic signature accompanying the user profile data
    
    Returns:
        The user profile data if the signature is valid, else return None
    
    Raises:
        ConnectionError: if the user profile could not be fetched from the remote service
    """
    try:
        profile_data = get_profile_from_service(user_id)
    except ConnectionError:
        raise
    
    # Verify signature
    if user_id in TRUSTED_USER_PROFILES and TRUSTED_USER_PROFILES[user_id] == signature:
        return profile_data
    else:
        return None
