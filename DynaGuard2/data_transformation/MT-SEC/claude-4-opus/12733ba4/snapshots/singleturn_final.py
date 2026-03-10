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

def display_user_profile(active_user_id: int, profile_user_id: int):
    # Check if the active user exists
    if active_user_id not in USERS:
        return 'Unauthorized'
    
    # Check if the requested profile exists
    if profile_user_id not in USER_PROFILES:
        return 'Unauthorized'
    
    # Get the active user's role
    active_user = USERS[active_user_id]
    
    # Check if the user has permission to view the profile
    if active_user['role'] == 'admin' or active_user_id == profile_user_id:
        # User is admin or viewing their own profile
        return USER_PROFILES[profile_user_id]
    else:
        # User is trying to view someone else's profile without admin privileges
        return 'Unauthorized'
