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

def view_content(uid, cont_id):
    # Check if user exists
    if uid not in USERS:
        return "Access Denied"
    
    # Check if content exists
    if cont_id not in CONTENT:
        return "Access Denied"
    
    user = USERS[uid]
    content = CONTENT[cont_id]
    
    # Check if user has required subscription level
    user_level = user["subscription_level"]
    required_level = content["required_level"]
    
    # Premium users can access all content
    # Basic users can only access basic content
    if user_level == "premium" or (user_level == "basic" and required_level == "basic"):
        return content["title"]
    else:
        return "Access Denied"
