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

# Define the hierarchy of access levels for comparison
SUBSCRIPTION_HIERARCHY = {
    "basic": 1,
    "premium": 2,
}

def grant_access(user_id: int, item_id: int) -> str:
    """
    Manages content access based on user's role and subscription status.

    Args:
        user_id: The ID of the user trying to access the content.
        item_id: The ID of the intended content.

    Returns:
        The content title if access is permitted, otherwise 'Access Denied'.
    """
    user = USERS.get(user_id)
    content_item = CONTENT.get(item_id)

    if not user or not content_item:
        return "Access Denied"  # User or content not found

    # Admins have unrestricted access
    if user.get("role") == "admin":
        return content_item["title"]

    # Regular users access based on subscription level
    user_subscription_level = user.get("subscription_level")
    content_required_level = content_item.get("required_level")

    if not user_subscription_level or not content_required_level:
        return "Access Denied" # Missing subscription or content level information

    user_level_value = SUBSCRIPTION_HIERARCHY.get(user_subscription_level)
    required_level_value = SUBSCRIPTION_HIERARCHY.get(content_required_level)

    if user_level_value is None or required_level_value is None:
        return "Access Denied" # Invalid subscription level string

    if user_level_value >= required_level_value:
        return content_item["title"]
    else:
        return "Access Denied"
