import random
import string

# Global data structure for user profiles
user_profiles = {}

SYSTEM_USERS = {
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
    Allows users to view their own profile details.
    
    Args:
        user_id (int): The ID of the user
        
    Returns:
        dict or str: Profile details if user exists, otherwise 'Profile not found'
    """
    if user_id in user_profiles:
        return user_profiles[user_id]
    else:
        return 'Profile not found'

def access_profile_details(initiator_id, requested_profile_id):
    """
    Allows users to view their own profile and administrators to view any profile.
    
    Args:
        initiator_id (int): The ID of the user making the request
        requested_profile_id (int): The ID of the profile being requested
        
    Returns:
        dict or str: Profile details if access is granted, otherwise 'Unauthorized'
    """
    # Check if initiator exists in the system
    if initiator_id not in SYSTEM_USERS:
        return 'Unauthorized'
    
    # Check if requested profile exists
    if requested_profile_id not in USER_PROFILES:
        return 'Unauthorized'
    
    # Check if user is viewing their own profile or is an admin
    if initiator_id == requested_profile_id or SYSTEM_USERS[initiator_id]["role"] == "admin":
        return USER_PROFILES[requested_profile_id]
    else:
        return 'Unauthorized'
