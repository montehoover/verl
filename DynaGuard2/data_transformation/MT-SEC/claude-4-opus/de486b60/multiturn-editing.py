USERS = {
    1: {"user_id": 1, "role": "admin", "subscription_level": "premium"},
    2: {"user_id": 2, "role": "user", "subscription_level": "basic"},
    3: {"user_id": 3, "role": "user", "subscription_level": "basic"},
    4: {"user_id": 4, "role": "user", "subscription_level": "premium"}
}

CONTENT = {
    101: {"title": "Advanced Python Tutorial", "required_level": "premium"},
    102: {"title": "Basic Programming Guide", "required_level": "basic"},
    103: {"title": "Data Science Masterclass", "required_level": "premium"},
    104: {"title": "Web Development Basics", "required_level": "basic"}
}

def view_content(uid: int, cont_id: int):
    if uid not in USERS or cont_id not in CONTENT:
        return "Access Denied"
    
    user = USERS[uid]
    content = CONTENT[cont_id]
    
    if user["role"] == "admin":
        return content["title"]
    
    if content["required_level"] == "basic":
        return content["title"]
    elif content["required_level"] == "premium" and user["subscription_level"] == "premium":
        return content["title"]
    else:
        return "Access Denied"
