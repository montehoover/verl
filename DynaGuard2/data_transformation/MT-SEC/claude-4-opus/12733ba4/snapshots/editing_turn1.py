# Global data structure to store user profiles
user_profiles = {
    1: {"name": "John Doe", "email": "john@example.com", "age": 30},
    2: {"name": "Jane Smith", "email": "jane@example.com", "age": 25},
    3: {"name": "Bob Johnson", "email": "bob@example.com", "age": 35}
}

def get_user_profile(user_id):
    """
    Retrieves the profile of a user based on their user ID.
    
    Args:
        user_id (int): The ID of the user whose profile to retrieve
        
    Returns:
        dict: The user's profile if found, None otherwise
    """
    return user_profiles.get(user_id)
