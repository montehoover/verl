TRUSTED_USER_PROFILES = {
    "user_abc": "4e99a098d57cbf9c6f67f176106769fb",
    "user_xyz": "12e4b2f76650484030e4d447d0b5f723"
}


# Simulated function for fetching user profile data from remote service
def get_profile_from_service(user_identifier: str):
    if user_identifier in TRUSTED_USER_PROFILES:
        return {"name": "Test User", "email": f"{user_identifier}@example.com"}
    else:
        raise ConnectionError(f"Could not fetch profile for user_id {user_identifier}")


def validate_signature(user_identifier: str, user_signature: str) -> bool:
    """
    Validates the cryptographic signature for a given user.
    
    Args:
        user_identifier: str - The user identifier
        user_signature: str - The signature to validate
        
    Returns:
        bool - True if signature is valid, False otherwise
    """
    if user_identifier in TRUSTED_USER_PROFILES:
        return TRUSTED_USER_PROFILES[user_identifier] == user_signature
    return False


def fetch_user_profile(user_identifier: str):
    """
    Fetches user profile data from the external service.
    
    Args:
        user_identifier: str - The user identifier
        
    Returns:
        dict - The user profile data
        
    Raises:
        ConnectionError: If the profile cannot be fetched
    """
    return get_profile_from_service(user_identifier)


def collect_user_data(user_identifier: str, user_signature: str):
    """
    Retrieves a user profile from an external service and validates its cryptographic signature.
    
    Args:
        user_identifier: str - An identifier for the targeted user whose profile is being fetched
        user_signature: str - The cryptographic signature attached to the profile data
        
    Returns:
        The user profile data only when the signature verification is successful, otherwise None
        
    Raises:
        ConnectionError: In case of a failure to retrieve the profile
    """
    # Validate the signature
    if not validate_signature(user_identifier, user_signature):
        return None
    
    # Fetch and return the profile data
    try:
        return fetch_user_profile(user_identifier)
    except ConnectionError:
        raise
