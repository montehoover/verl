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
    # Check if active user exists
    if active_user_id not in USERS:
        return "Unauthorized"
    
    # Check if requested profile exists
    if profile_user_id not in USER_PROFILES:
        return "Unauthorized"
    
    # Get the active user's role
    active_user = USERS[active_user_id]
    
    # Allow access if user is viewing their own profile or if they are an admin
    if active_user_id == profile_user_id or active_user["role"] == "admin":
        return USER_PROFILES[profile_user_id]
    
    return "Unauthorized"
