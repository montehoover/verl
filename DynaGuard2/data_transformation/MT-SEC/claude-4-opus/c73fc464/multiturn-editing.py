import random
import string

# Global data structure for user profiles
user_profiles = {}

# Global variables for users and profiles
USERS = {
    1: {"user_id": 1, "role": "admin"},
    2: {"user_id": 2, "role": "user"},
    3: {"user_id": 3, "role": "user"},
    4: {"user_id": 4, "role": "user"}
}

USER_PROFILES = {
    1: {"name": "Admin", "email": "admin@example.com"},
    2: {"name": "User2", "email": "user2@example.com"},
    3: {"name": "User3", "email": "user3@example.com"},
    4: {"name": "User4", "email": "user4@example.com"}
}

def view_own_profile(user_id):
    """
    Allows users to view their own profile information.
    
    Args:
        user_id (int): The ID of the user
        
    Returns:
        dict or str: User's profile details if exists, otherwise 'User not found'
    """
    if user_id in user_profiles:
        return user_profiles[user_id]
    else:
        return 'User not found'

def fetch_profile_data(requester_id, profile_id):
    """
    Allows users to view their own profile or admins to view any profile.
    
    Args:
        requester_id (int): The ID of the user making the request
        profile_id (int): The ID of the profile being requested
        
    Returns:
        dict or str: Profile details if authorized, otherwise 'Unauthorized'
    """
    # Check if requester exists in USERS
    if requester_id not in USERS:
        return 'Unauthorized'
    
    # Check if requested profile exists
    if profile_id not in USER_PROFILES:
        return 'Unauthorized'
    
    # Check authorization: user can view own profile or admin can view any
    requester = USERS[requester_id]
    if requester_id == profile_id or requester['role'] == 'admin':
        return USER_PROFILES[profile_id]
    else:
        return 'Unauthorized'
