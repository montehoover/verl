# Sample user profiles dictionary
user_profiles = {
    "user123": {"name": "John Doe", "email": "john.doe@example.com"},
    "user456": {"name": "Jane Smith", "email": "jane.smith@example.com"},
    "user789": {"name": "Bob Johnson", "email": "bob.johnson@example.com"}
}

# List of trusted user IDs
trusted_user_ids = ["user123", "user456", "user789"]

def get_user_profile(user_id):
    """
    Retrieves a user profile from the local dictionary.
    
    Args:
        user_id (str): The user ID to look up
        
    Returns:
        dict: A dictionary with user's name and email if found and trusted, None otherwise
    """
    # Check if user_id is in the trusted list
    if user_id not in trusted_user_ids:
        return None
    
    return user_profiles.get(user_id)
