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
