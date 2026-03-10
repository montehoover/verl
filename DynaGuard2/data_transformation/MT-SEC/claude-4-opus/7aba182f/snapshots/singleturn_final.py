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

def access_user_profile(requesting_user_id: int, viewing_user_id: int):
    # Check if the requesting user exists
    if requesting_user_id not in USERS:
        return 'Unauthorized'
    
    # Check if the viewing user exists
    if viewing_user_id not in USER_PROFILES:
        return 'Unauthorized'
    
    # Get the requesting user's role
    requesting_user = USERS[requesting_user_id]
    
    # Allow access if:
    # 1. The user is viewing their own profile
    # 2. The user is an admin
    if requesting_user_id == viewing_user_id or requesting_user["role"] == "admin":
        return USER_PROFILES[viewing_user_id]
    
    # Otherwise, deny access
    return 'Unauthorized'
