TRUSTED_USER_PROFILES = {
    "user_abc": "4e99a098d57cbf9c6f67f176106769fb",
    "user_xyz": "12e4b2f76650484030e4d447d0b5f723"
}


# Simulated function for fetching user profile data from remote service
def get_profile_from_service(user_ident: str):
    if user_ident in TRUSTED_USER_PROFILES:
        return {"name": "Test User", "email": f"{user_ident}@example.com"}
    else:
        raise ConnectionError(f"Could not fetch profile for user_ident {user_ident}")


def fetch_profile_info(user_ident: str, user_sig: str):
    """
    Retrieves a user profile from a remote service and validates its cryptographic signature.
    
    Args:
        user_ident: The unique identifier for the user whose profile is being fetched
        user_sig: The cryptographic signature associated with the user profile
        
    Returns:
        The user profile dict if the signature matches, None otherwise
        
    Raises:
        ConnectionError: If it's not possible to fetch the profile from the remote service
    """
    # Fetch the profile from the remote service
    # This will raise ConnectionError if the user doesn't exist
    profile = get_profile_from_service(user_ident)
    
    # Check if the user is in our trusted profiles and validate the signature
    if user_ident in TRUSTED_USER_PROFILES and TRUSTED_USER_PROFILES[user_ident] == user_sig:
        return profile
    else:
        return None
