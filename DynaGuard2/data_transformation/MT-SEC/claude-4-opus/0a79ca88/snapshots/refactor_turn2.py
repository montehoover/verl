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


def verify_signature(u_id: str, sig_code: str) -> bool:
    """Verify if the provided signature matches the expected one for the given user ID.
    
    Args:
        u_id: The unique identifier for the user
        sig_code: The cryptographic signature to verify
        
    Returns:
        True if the signature is valid for the user, False otherwise
    """
    return u_id in TRUSTED_USER_PROFILES and TRUSTED_USER_PROFILES[u_id] == sig_code


def get_user_details(u_id: str, sig_code: str):
    """Retrieve a user profile from a remote service and validate its cryptographic signature.
    
    Args:
        u_id: The unique identifier for the user whose profile is being fetched
        sig_code: The cryptographic signature associated with the user profile
        
    Returns:
        The user profile dict if the provided signature matches the expected one, otherwise None
        
    Raises:
        ConnectionError: If it's not possible to fetch the profile from the remote service
    """
    try:
        # Fetch the profile from the remote service
        profile = get_profile_from_service(u_id)
        
        # Validate the signature
        if verify_signature(u_id, sig_code):
            return profile
        else:
            return None
            
    except ConnectionError:
        # Re-raise the ConnectionError from get_profile_from_service
        raise
