MEMBER_DATA = {
    1: {"user_id": 1, "role": "admin"},
    2: {"user_id": 2, "role": "user"},
    3: {"user_id": 3, "role": "user"},
    4: {"user_id": 4, "role": "user"}
}

MEMBER_PROFILES = {
    1: {"name": "Admin", "email": "admin@example.com"},
    2: {"name": "User2", "email": "user2@example.com"},
    3: {"name": "User3", "email": "user3@example.com"},
    4: {"name": "User4", "email": "user4@example.com"}
}

def get_user_role(user_id):
    """
    Retrieves the role of a user based on their user_id.
    
    Args:
        user_id (int): The ID of the user
        
    Returns:
        str: The role of the user, or None if user not found
    """
    user_data = MEMBER_DATA.get(user_id)
    if user_data:
        return user_data["role"]
    return None

def can_access_profile(requester_id, profile_id):
    """
    Checks if a user can access another user's profile.
    
    Args:
        requester_id (int): The ID of the user requesting access
        profile_id (int): The ID of the profile being accessed
        
    Returns:
        bool: True if access is allowed, False otherwise
    """
    # Users can always access their own profile
    if requester_id == profile_id:
        return True
    
    # Get the requester's role
    requester_role = get_user_role(requester_id)
    
    # If requester doesn't exist, deny access
    if requester_role is None:
        return False
    
    # Admins can access any profile
    if requester_role == "admin":
        return True
    
    # Regular users can only access their own profile
    return False

def fetch_profile_data(requester_id, profile_id):
    """
    Fetches user profile data if the requester has access.
    
    Args:
        requester_id (int): The ID of the user requesting the profile
        profile_id (int): The ID of the profile to fetch
        
    Returns:
        dict or str: Profile data if access is granted, 'Unauthorized' otherwise
    """
    # Check if the requester has access to the profile
    if can_access_profile(requester_id, profile_id):
        # Return the profile data if it exists
        profile_data = MEMBER_PROFILES.get(profile_id)
        if profile_data:
            return profile_data
        else:
            return 'Unauthorized'
    else:
        return 'Unauthorized'
