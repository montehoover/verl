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

def grant_access(user_id: int, item_id: int):
    # Check if user exists
    if user_id not in USERS:
        return "Access Denied"
    
    # Check if content exists
    if item_id not in CONTENT:
        return "Access Denied"
    
    user = USERS[user_id]
    content = CONTENT[item_id]
    
    # Admins have unrestricted access
    if user["role"] == "admin":
        return content["title"]
    
    # Regular users need matching subscription level
    if user["role"] == "user":
        # Basic users can only access basic content
        if user["subscription_level"] == "basic" and content["required_level"] == "basic":
            return content["title"]
        # Premium users can access both basic and premium content
        elif user["subscription_level"] == "premium":
            return content["title"]
    
    return "Access Denied"
