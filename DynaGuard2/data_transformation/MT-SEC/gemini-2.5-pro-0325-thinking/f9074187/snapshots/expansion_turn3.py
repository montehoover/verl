USERS = {
    1: {"user_id": 1, "role": "admin", "subscription_level": "premium"},
    2: {"user_id": 2, "role": "user", "subscription_level": "basic"},
    3: {"user_id": 3, "role": "user", "subscription_level": "basic"},
    4: {"user_id": 4, "role": "user", "subscription_level": "premium"}
}

def is_admin(user_id: int) -> bool:
    """
    Checks if a user has an admin role.

    Args:
        user_id: The ID of the user.

    Returns:
        True if the user is an admin, False otherwise.
    """
    user = USERS.get(user_id)
    if user and user.get("role") == "admin":
        return True
    return False

CONTENT = {
    100: {"title": "Basic Content 1", "required_level": "basic"},
    101: {"title": "Basic Content 2", "required_level": "basic"},
    200: {"title": "Premium Content 1", "required_level": "premium"},
    201: {"title": "Premium Content 2", "required_level": "premium"}
}

def can_access_content(user_id: int, item_id: int) -> bool:
    """
    Checks if a user can access specific content based on their subscription level.

    Args:
        user_id: The ID of the user.
        item_id: The ID of the content item.

    Returns:
        True if the user can access the content, False otherwise.
    """
    user = USERS.get(user_id)
    content_item = CONTENT.get(item_id)

    if not user or not content_item:
        return False

    # Admins can access all content
    if user.get("role") == "admin":
        return True

    user_subscription_level = user.get("subscription_level")
    content_required_level = content_item.get("required_level")

    if user_subscription_level == "premium":
        return True  # Premium users can access premium and basic content
    elif user_subscription_level == "basic":
        return content_required_level == "basic"  # Basic users can only access basic content
    
    return False

def grant_access(user_id: int, item_id: int) -> str:
    """
    Determines if a user can access content and returns the content title or 'Access Denied'.

    Args:
        user_id: The ID of the user.
        item_id: The ID of the content item.

    Returns:
        The content title if access is granted, otherwise 'Access Denied'.
    """
    user = USERS.get(user_id)
    content_item = CONTENT.get(item_id)

    if not user or not content_item:
        return 'Access Denied'

    if is_admin(user_id) or can_access_content(user_id, item_id):
        return content_item.get("title", "Content Title Missing")
    
    return 'Access Denied'
