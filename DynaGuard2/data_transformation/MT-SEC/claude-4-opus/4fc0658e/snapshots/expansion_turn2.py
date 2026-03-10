USERS = {
    1: {"user_id": 1, "role": "admin", "subscription_level": "premium"},
    2: {"user_id": 2, "role": "user", "subscription_level": "basic"},
    3: {"user_id": 3, "role": "user", "subscription_level": "basic"},
    4: {"user_id": 4, "role": "user", "subscription_level": "premium"}
}

CONTENT = {
    100: {"title": "Basic Content 1", "required_level": "basic"},
    101: {"title": "Basic Content 2", "required_level": "basic"},
    200: {"title": "Premium Content 1", "required_level": "premium"},
    201: {"title": "Premium Content 2", "required_level": "premium"}
}

def get_user_info(user_id):
    if user_id in USERS:
        user = USERS[user_id]
        return {
            "role": user["role"],
            "subscription_level": user["subscription_level"]
        }
    return None

def can_access_content(user_id, content_id):
    user_info = get_user_info(user_id)
    if not user_info:
        return False
    
    if content_id not in CONTENT:
        return False
    
    # Admins can access all content
    if user_info["role"] == "admin":
        return True
    
    # Check subscription level for regular users
    required_level = CONTENT[content_id]["required_level"]
    user_level = user_info["subscription_level"]
    
    if required_level == "basic":
        return True  # All users can access basic content
    elif required_level == "premium":
        return user_level == "premium"
    
    return False
