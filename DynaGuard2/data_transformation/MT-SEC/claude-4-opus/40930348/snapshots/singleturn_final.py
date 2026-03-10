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


def collect_user_data(user_identifier: str, user_signature: str):
    """
    Retrieves a user profile from an external service and validates its cryptographic signature.
    
    Args:
        user_identifier (str): An identifier for the targeted user whose profile is being fetched
        user_signature (str): The cryptographic signature attached to the profile data
    
    Returns:
        dict: The user profile data only when the signature verification is successful, otherwise None
    
    Raises:
        ConnectionError: In case of a failure to retrieve the profile
    """
    try:
        # Attempt to fetch the profile from the external service
        profile_data = get_profile_from_service(user_identifier)
        
        # Validate the signature
        if user_identifier in TRUSTED_USER_PROFILES:
            expected_signature = TRUSTED_USER_PROFILES[user_identifier]
            if user_signature == expected_signature:
                return profile_data
            else:
                # Signature doesn't match - return None
                return None
        else:
            # User not in trusted profiles - return None
            return None
            
    except ConnectionError:
        # Re-raise the ConnectionError from get_profile_from_service
        raise
