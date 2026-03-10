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


def fetch_content(given_user_id: int, target_content_id: int) -> str:
    """
    Manages content access based on user role and subscription.

    Rules:
    - Admins have unrestricted access to any existing content.
    - Regular users can access content if their subscription level meets the
      required level for that content. Assumption: premium >= basic.

    Args:
        given_user_id: ID of the user trying to access content.
        target_content_id: ID of the content being accessed.

    Returns:
        The content title if access is permitted; otherwise, "Access Denied".
    """
    user = USERS.get(given_user_id)
    content = CONTENT.get(target_content_id)

    # Validate existence of user and content
    if not user or not content:
        return "Access Denied"

    # Admins have full access
    if user.get("role") == "admin":
        return content.get("title", "Access Denied")

    # Enforce subscription level for regular users
    user_sub_level = user.get("subscription_level")
    required_level = content.get("required_level")
    if not user_sub_level or not required_level:
        return "Access Denied"

    # Define a simple hierarchy where premium >= basic
    hierarchy = {"basic": 0, "premium": 1}

    # If levels are unknown, deny access
    if user_sub_level not in hierarchy or required_level not in hierarchy:
        return "Access Denied"

    # Allow if user's level meets or exceeds requirement
    if hierarchy[user_sub_level] >= hierarchy[required_level]:
        return content.get("title", "Access Denied")

    return "Access Denied"
