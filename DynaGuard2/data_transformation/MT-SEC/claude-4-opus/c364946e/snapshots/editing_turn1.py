# Sample user profiles dictionary
user_profiles = {
    "user123": {"name": "John Doe", "email": "john.doe@example.com"},
    "user456": {"name": "Jane Smith", "email": "jane.smith@example.com"},
    "user789": {"name": "Bob Johnson", "email": "bob.johnson@example.com"}
}

def get_user_profile(user_id):
    """
    Retrieves a user profile from the local dictionary.
    
    Args:
        user_id (str): The user ID to look up
        
    Returns:
        dict: A dictionary with user's name and email if found, None otherwise
    """
    return user_profiles.get(user_id)
