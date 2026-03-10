# Trusted user profiles with their cryptographic signatures
TRUSTED_USER_PROFILES = {
    "user_abc": "4e99a098d57cbf9c6f67f176106769fb",
    "user_xyz": "12e4b2f76650484030e4d447d0b5f723"
}

def get_profile_from_service(user_id: str):
    if user_id in TRUSTED_USER_PROFILES:
        return {"name": "Test User", "email": f"{user_id}@example.com"}
    else:
        raise ConnectionError(f"Could not fetch profile for user_id {user_id}")

def fetch_user_profile(user_id: str, signature: str):
    """
    Retrieves a user profile from a remote service and verifies it using a cryptographic signature.
    
    Args:
        user_id (str): The unique identifier of the user
        signature (str): The cryptographic signature
        
    Returns:
        dict: The profile data if the signature is valid, None otherwise
    """
    # Check if the user_id exists in trusted profiles
    if user_id not in TRUSTED_USER_PROFILES:
        return None
    
    # Verify the signature
    if TRUSTED_USER_PROFILES[user_id] != signature:
        return None
    
    # Fetch the profile from the service
    try:
        profile = get_profile_from_service(user_id)
        return profile
    except ConnectionError:
        return None
