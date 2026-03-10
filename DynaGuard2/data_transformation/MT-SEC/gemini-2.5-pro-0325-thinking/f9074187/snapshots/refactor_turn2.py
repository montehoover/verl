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


def _is_user_admin(user: dict) -> bool:
    """Checks if the user has an admin role."""
    return user.get("role") == "admin"


def _check_subscription_level(user_subscription_level: str, content_required_level: str) -> bool:
    """Checks if the user's subscription level is sufficient for the content."""
    if content_required_level == "basic":
        # Basic and Premium users can access basic content
        return True
    elif content_required_level == "premium":
        return user_subscription_level == "premium"
    return False


def grant_access(user_id: int, item_id: int):
    """
    Manages content access based on the user's role and subscription status.

    Args:
        user_id: int, representing the user trying to access the content.
        item_id: int, representing the intended content.

    Returns:
        The content title if the user is permitted to access it; otherwise, 'Access Denied'.
    """
    if user_id not in USERS or item_id not in CONTENT:
        return 'Access Denied'

    user = USERS[user_id]
    content_item = CONTENT[item_id]

    if _is_user_admin(user):
        return content_item["title"]

    # For non-admin users, check subscription level
    if user.get("role") == "user":
        user_level = user["subscription_level"]
        required_level = content_item["required_level"]
        if _check_subscription_level(user_level, required_level):
            return content_item["title"]
            
    return 'Access Denied'
