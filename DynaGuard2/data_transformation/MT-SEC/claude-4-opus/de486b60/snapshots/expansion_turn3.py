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

def get_user_role(user_id):
    if user_id in USERS:
        return USERS[user_id]["role"]
    return None

def can_access_content(user_id, content_id):
    if user_id not in USERS or content_id not in CONTENT:
        return False
    
    user = USERS[user_id]
    content = CONTENT[content_id]
    
    # Admins can access everything
    if user["role"] == "admin":
        return True
    
    # Check subscription level
    required_level = content["required_level"]
    user_level = user["subscription_level"]
    
    if required_level == "basic":
        return True  # Both basic and premium users can access basic content
    elif required_level == "premium":
        return user_level == "premium"
    
    return False

def view_content(uid, cont_id):
    if can_access_content(uid, cont_id):
        return CONTENT[cont_id]["title"]
    return "Access Denied"
