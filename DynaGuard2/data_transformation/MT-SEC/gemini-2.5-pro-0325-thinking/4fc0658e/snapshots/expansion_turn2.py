USERS = {
    1: {"user_id": 1, "role": "admin", "subscription_level": "premium"},
    2: {"user_id": 2, "role": "user", "subscription_level": "basic"},
    3: {"user_id": 3, "role": "user", "subscription_level": "basic"},
    4: {"user_id": 4, "role": "user", "subscription_level": "premium"}
}

def get_user_info(user_id: int) -> dict:
    """
    Retrieves a user's role and subscription level.

    Args:
        user_id: The ID of the user.

    Returns:
        A dictionary with keys 'role' and 'subscription_level',
        or an empty dictionary if the user is not found.
    """
    user_data = USERS.get(user_id)
    if user_data:
        return {
            "role": user_data.get("role"),
            "subscription_level": user_data.get("subscription_level")
        }
    return {}

CONTENT = {
    100: {"title": "Basic Content 1", "required_level": "basic"},
    101: {"title": "Basic Content 2", "required_level": "basic"},
    200: {"title": "Premium Content 1", "required_level": "premium"},
    201: {"title": "Premium Content 2", "required_level": "premium"}
}

def can_access_content(user_id: int, content_id: int) -> bool:
    """
    Checks if a user can access specific content.

    Args:
        user_id: The ID of the user.
        content_id: The ID of the content.

    Returns:
        True if the user can access the content, False otherwise.
    """
    user_info = get_user_info(user_id)
    content_info = CONTENT.get(content_id)

    if not user_info or not content_info:
        return False  # User or content not found

    user_role = user_info.get("role")
    user_subscription = user_info.get("subscription_level")
    content_required_level = content_info.get("required_level")

    if user_role == "admin":
        return True  # Admins can access all content

    if content_required_level == "basic":
        return True # All authenticated users (basic or premium) can access basic content
    
    if content_required_level == "premium":
        return user_subscription == "premium" # Only premium users can access premium content

    return False # Default to no access
