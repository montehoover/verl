# User profiles database
user_profiles = {
    "user123": {"name": "John Doe", "email": "john.doe@example.com"},
    "user456": {"name": "Jane Smith", "email": "jane.smith@example.com"},
    "user789": {"name": "Bob Johnson", "email": "bob.johnson@example.com"}
}

def get_user_profile(uid, trusted_ids):
    """
    Retrieves a user profile from the local dictionary.
    
    Args:
        uid (str): The user ID to look up
        trusted_ids (list): List of trusted user IDs
        
    Returns:
        dict: A dictionary with user's name and email if found and trusted, None otherwise
    """
    if uid not in trusted_ids:
        return None
    return user_profiles.get(uid)

TRUSTED_USER_PROFILES = {
    "user_abc": "4e99a098d57cbf9c6f67f176106769fb",
    "user_xyz": "12e4b2f76650484030e4d447d0b5f723"
}

def get_profile_from_service(uid: str):
    if uid in TRUSTED_USER_PROFILES:
        return {"name": "Test Individual", "email": f"{uid}@example.com"}
    else:
        raise ConnectionError(f"Could not fetch profile for uid {uid}")

def retrieve_user_profile(uid, sig):
    """
    Retrieves a user profile from a remote service and validates it using a cryptographic signature.
    
    Args:
        uid (str): The unique identifier for the user
        sig (str): The cryptographic signature
        
    Returns:
        dict: The user profile if signature is valid, None otherwise
    """
    if uid not in TRUSTED_USER_PROFILES:
        return None
    
    if TRUSTED_USER_PROFILES[uid] != sig:
        return None
    
    try:
        return get_profile_from_service(uid)
    except ConnectionError:
        return None
