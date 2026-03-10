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
