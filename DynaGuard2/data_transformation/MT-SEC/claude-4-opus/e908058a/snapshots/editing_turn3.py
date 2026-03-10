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

def get_user_profile(user_id, trusted_ids):
    """
    Retrieves a user profile from the local dictionary based on user ID.
    
    Args:
        user_id (str): The ID of the user to retrieve
        trusted_ids (list): List of trusted user IDs
        
    Returns:
        dict: Dictionary containing user's name and email, or None if not found or not trusted
    """
    if user_id not in trusted_ids:
        return None
    return user_profiles.get(user_id)

TRUSTED_USER_PROFILES = {
    "user_abc": "4e99a098d57cbf9c6f67f176106769fb",
    "user_xyz": "12e4b2f76650484030e4d447d0b5f723"
}

def get_profile_from_service(uuid: str):
    if uuid in TRUSTED_USER_PROFILES:
        return {"name": "Test User", "email": f"{uuid}@example.com"}
    else:
        raise ConnectionError(f"Could not fetch profile for user_id {uuid}")

def collect_user_data(uuid, sig_hash):
    """
    Fetches a user profile from a remote system and verifies it using a cryptographic signature.
    
    Args:
        uuid (str): The unique identifier for the user
        sig_hash (str): The cryptographic signature
        
    Returns:
        dict: Profile data if signature is valid, otherwise None
        
    Raises:
        ConnectionError: If there's a connectivity issue
    """
    # Check if the signature is valid
    if uuid not in TRUSTED_USER_PROFILES or TRUSTED_USER_PROFILES[uuid] != sig_hash:
        return None
    
    # Fetch the profile from the service
    return get_profile_from_service(uuid)
