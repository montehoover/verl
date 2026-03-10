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

def fetch_content(given_user_id: int, target_content_id: int) -> str:
    # Check if user exists
    if given_user_id not in USERS:
        return "Access Denied"
    
    # Check if content exists
    if target_content_id not in CONTENT:
        return "Access Denied"
    
    user = USERS[given_user_id]
    content = CONTENT[target_content_id]
    
    # Admins have unrestricted access
    if user["role"] == "admin":
        return content["title"]
    
    # Regular users need matching or higher subscription level
    if user["role"] == "user":
        user_level = user["subscription_level"]
        required_level = content["required_level"]
        
        # Check subscription level compatibility
        if user_level == "premium" or (user_level == "basic" and required_level == "basic"):
            return content["title"]
    
    return "Access Denied"
