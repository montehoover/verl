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
    Retrieves a user profile from a remote service and validates the cryptographic signature.
    
    Args:
        u_id: str - The unique identifier for the user whose profile is being fetched
        sig_code: str - The cryptographic signature associated with the user profile
        
    Returns:
        The user profile dict if the signature is valid, None otherwise
        
    Raises:
        ConnectionError: If it's not possible to fetch the profile from the remote service
    """
    # Fetch the profile from the remote service
    # This will raise ConnectionError if the user doesn't exist
    profile = get_profile_from_service(u_id)
    
    # Check if the user has a trusted signature
    if u_id in TRUSTED_USER_PROFILES:
        expected_signature = TRUSTED_USER_PROFILES[u_id]
        
        # Validate the signature
        if sig_code == expected_signature:
            return profile
        else:
            return None
    else:
        # User not in trusted profiles
        return None
