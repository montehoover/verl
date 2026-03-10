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

def is_admin(user_id):
    user = USERS.get(user_id)
    if user:
        return user["role"] == "admin"
    return False

def can_access_content(user_id, item_id):
    user = USERS.get(user_id)
    content = CONTENT.get(item_id)
    
    if not user or not content:
        return False
    
    user_level = user["subscription_level"]
    required_level = content["required_level"]
    
    if user_level == "premium":
        return True
    elif user_level == "basic" and required_level == "basic":
        return True
    else:
        return False

def grant_access(user_id, item_id):
    if is_admin(user_id) or can_access_content(user_id, item_id):
        content = CONTENT.get(item_id)
        if content:
            return content["title"]
    return "Access Denied"
