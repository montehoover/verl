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
    Admins can access any content, while users can only access content
    that matches their subscription.

    Args:
        current_user_id: The user_id of the actor attempting to access content.
        content_id: The identifier of the content being accessed.

    Returns:
        The content data (title) if the user is authorized to view it,
        else 'Access Denied'.
    """
    user = USERS.get(current_user_id)
    content = CONTENT.get(content_id)

    if not user or not content:
        return "Access Denied"

    user_role = user.get("role")
    user_subscription = user.get("subscription_level")
    content_required_level = content.get("required_level")

    if user_role == "admin":
        return content.get("title")

    if user_subscription == "premium":
        # Premium users can access premium and basic content
        if content_required_level == "premium" or content_required_level == "basic":
            return content.get("title")
    elif user_subscription == "basic":
        # Basic users can only access basic content
        if content_required_level == "basic":
            return content.get("title")

    return "Access Denied"
