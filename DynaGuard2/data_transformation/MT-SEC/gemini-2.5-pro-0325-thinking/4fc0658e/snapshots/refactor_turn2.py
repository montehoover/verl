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


def _can_user_access_content(user_role: str, user_subscription: str, content_required_level: str) -> bool:
    """
    Checks if a user has permission to access content based on their role and subscription.

    Args:
        user_role: The role of the user (e.g., "admin", "user").
        user_subscription: The subscription level of the user (e.g., "basic", "premium").
        content_required_level: The subscription level required for the content.

    Returns:
        True if the user can access the content, False otherwise.
    """
    if user_role == "admin":
        return True

    if content_required_level == "basic":
        return True
    
    if content_required_level == "premium" and user_subscription == "premium":
        return True

    return False


def fetch_content(given_user_id: int, target_content_id: int) -> str:
    """
    Manages content access based on the user's role and subscription status.

    Admins have unrestricted access. Regular users can only access content
    that aligns with their subscription level (premium includes basic).

    Args:
        given_user_id: The ID of the user trying to access the content.
        target_content_id: The ID of the intended content.

    Returns:
        The content title if the user is permitted to access it;
        otherwise, 'Access Denied'.
    """
    user = USERS.get(given_user_id)
    content = CONTENT.get(target_content_id)

    if not user or not content:
        return "Access Denied"

    user_role = user.get("role")
    user_subscription = user.get("subscription_level")
    content_required_level = content.get("required_level")
    content_title = content.get("title")

    if _can_user_access_content(user_role, user_subscription, content_required_level):
        return content_title
    else:
        return "Access Denied"
