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

def access_content(current_user_id: int, content_id: int):
    """
    Allows users to access content based on their subscription level.
    Admins can access any content.
    Users can only access content that matches their subscription level,
    with premium users also having access to basic content.

    Args:
        current_user_id: The ID of the user attempting to access content.
        content_id: The ID of the content being accessed.

    Returns:
        The content data if the user is authorized, otherwise 'Access Denied'.
    """
    user = USERS.get(current_user_id)
    content_item = CONTENT.get(content_id)

    if not user or not content_item:
        return "Access Denied"

    user_role = user.get("role")
    user_subscription = user.get("subscription_level")
    content_required_level = content_item.get("required_level")

    if user_role == "admin":
        return content_item

    if user_role == "user":
        if user_subscription == "premium":
            # Premium users can access premium and basic content
            if content_required_level in ["premium", "basic"]:
                return content_item
        elif user_subscription == "basic":
            # Basic users can only access basic content
            if content_required_level == "basic":
                return content_item
    
    return "Access Denied"
