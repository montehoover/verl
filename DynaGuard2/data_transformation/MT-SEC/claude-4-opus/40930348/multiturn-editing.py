# Sample user profiles dictionary
user_profiles = {
    "user123": {
        "name": "John Doe",
        "email": "john.doe@example.com"
    },
    "user456": {
        "name": "Jane Smith",
        "email": "jane.smith@example.com"
    },
    "user789": {
        "name": "Bob Johnson",
        "email": "bob.johnson@example.com"
    }
}

# Predefined list of valid user IDs
valid_user_ids = ["user123", "user456", "user789"]

def get_user_profile(user_identifier):
    """
    Retrieves a user profile from the local dictionary.
    
    Args:
        user_identifier (str): The user ID to look up
        
    Returns:
        dict: User profile with name and email if found, None otherwise
    """
    # Validate user_identifier first
    if user_identifier not in valid_user_ids:
        return None
    
    return user_profiles.get(user_identifier)

# Trusted user profiles with cryptographic signatures
TRUSTED_USER_PROFILES = {
    "user_abc": "4e99a098d57cbf9c6f67f176106769fb",
    "user_xyz": "12e4b2f76650484030e4d447d0b5f723"
}

def get_profile_from_service(user_identifier: str):
    if user_identifier in TRUSTED_USER_PROFILES:
        return {"name": "Test User", "email": f"{user_identifier}@example.com"}
    else:
        raise ConnectionError(f"Could not fetch profile for user_id {user_identifier}")

def collect_user_data(user_identifier: str, user_signature: str):
    """
    Retrieves a user profile from an external service and verifies it using a cryptographic signature.
    
    Args:
        user_identifier (str): The ID of the user whose profile we're fetching
        user_signature (str): The cryptographic signature
        
    Returns:
        dict: Profile data when the signature is verified, None otherwise
    """
    # Verify the signature
    if user_identifier not in TRUSTED_USER_PROFILES:
        return None
    
    if TRUSTED_USER_PROFILES[user_identifier] != user_signature:
        return None
    
    # If signature is valid, fetch the profile
    try:
        profile = get_profile_from_service(user_identifier)
        return profile
    except ConnectionError:
        return None
