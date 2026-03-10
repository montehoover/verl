# Global data structure to store user profiles
user_profiles = {
    1: {"name": "John Doe", "email": "john@example.com", "age": 30, "role": "admin"},
    2: {"name": "Jane Smith", "email": "jane@example.com", "age": 25, "role": "user"},
    3: {"name": "Bob Johnson", "email": "bob@example.com", "age": 35, "role": "user"}
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

def view_profile_with_access(active_user_id, profile_user_id):
    """
    Allows admins to view any user's profile, while regular users can only view their own.
    
    Args:
        active_user_id (int): The ID of the user making the request
        profile_user_id (int): The ID of the user whose profile is being requested
        
    Returns:
        dict or str: The user's profile if access is allowed, 'Unauthorized' otherwise
    """
    active_user = get_user_profile(active_user_id)
    
    if not active_user:
        return "Unauthorized"
    
    # Admins can view any profile
    if active_user.get("role") == "admin":
        return get_user_profile(profile_user_id)
    
    # Regular users can only view their own profile
    if active_user_id == profile_user_id:
        return get_user_profile(profile_user_id)
    
    return "Unauthorized"
