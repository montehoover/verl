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
    # Guard clause: Check if user exists
    if uid not in USERS:
        return "Access Denied"
    
    # Guard clause: Check if content exists
    if cont_id not in CONTENT:
        return "Access Denied"
    
    user = USERS[uid]
    content = CONTENT[cont_id]
    
    # Guard clause: Admin users can access all content
    if user["role"] == "admin":
        return content["title"]
    
    # Guard clause: Check subscription level
    if user["subscription_level"] == "premium":
        return content["title"]
    
    if user["subscription_level"] == "basic" and content["required_level"] == "basic":
        return content["title"]
    
    return "Access Denied"
