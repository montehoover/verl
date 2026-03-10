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

def is_admin(user: dict) -> bool:
    """Check if the user has admin role."""
    return user.get("role") == "admin"

def has_sufficient_subscription(user_level: str, required_level: str) -> bool:
    """Check if user's subscription level meets the content requirement."""
    if user_level == "premium":
        return True
    return user_level == "basic" and required_level == "basic"

def can_access_content(user: dict, content: dict) -> bool:
    """Determine if a user can access specific content based on role and subscription."""
    if is_admin(user):
        return True
    
    if user.get("role") == "user":
        return has_sufficient_subscription(
            user.get("subscription_level"),
            content.get("required_level")
        )
    
    return False

def fetch_content(given_user_id: int, target_content_id: int) -> str:
    # Check if user exists
    if given_user_id not in USERS:
        return "Access Denied"
    
    # Check if content exists
    if target_content_id not in CONTENT:
        return "Access Denied"
    
    user = USERS[given_user_id]
    content = CONTENT[target_content_id]
    
    # Check access permissions
    if can_access_content(user, content):
        return content["title"]
    
    return "Access Denied"
