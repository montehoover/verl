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

def is_admin(user):
    """Check if user has admin role."""
    return user.get("role") == "admin"

def has_subscription_access(user, content):
    """Check if user's subscription level meets content requirements."""
    user_level = user.get("subscription_level")
    required_level = content.get("required_level")
    
    # Premium users can access all content
    if user_level == "premium":
        return True
    
    # Basic users can only access basic content
    return user_level == "basic" and required_level == "basic"

def can_access_content(user, content):
    """Determine if user can access the content based on role and subscription."""
    # Admins have unrestricted access
    if is_admin(user):
        return True
    
    # Regular users need appropriate subscription level
    if user.get("role") == "user":
        return has_subscription_access(user, content)
    
    return False

def grant_access(user_id: int, item_id: int):
    # Check if user exists
    if user_id not in USERS:
        return "Access Denied"
    
    # Check if content exists
    if item_id not in CONTENT:
        return "Access Denied"
    
    user = USERS[user_id]
    content = CONTENT[item_id]
    
    if can_access_content(user, content):
        return content["title"]
    
    return "Access Denied"
