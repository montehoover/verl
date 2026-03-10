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

_LEVEL_RANK = {
    "basic": 0,
    "premium": 1,
}


def is_admin(user: dict) -> bool:
    """
    Pure function to check if a user has an admin role.
    """
    return user.get("role") == "admin"


def meets_required_level(user_level: str, required_level: str) -> bool:
    """
    Pure function to check if a user's subscription level meets or exceeds
    the required level for content.
    """
    try:
        return _LEVEL_RANK[user_level] >= _LEVEL_RANK[required_level]
    except KeyError:
        return False


def can_user_access_content(user: dict, content: dict) -> bool:
    """
    Pure function that determines if a user can access a piece of content
    based on role and subscription level.
    """
    if is_admin(user):
        return True

    user_level = user.get("subscription_level")
    required_level = content.get("required_level")
    return meets_required_level(user_level, required_level)


def grant_access(user_id: int, item_id: int) -> str:
    """
    Manages content access based on the user's role and subscription status.

    - Admins: Unrestricted access to any content.
    - Regular users: Access allowed only if their subscription level meets or exceeds
      the content's required level (e.g., premium can access basic and premium).

    Args:
        user_id: The user trying to access the content.
        item_id: The intended content.

    Returns:
        The content title if permitted; otherwise, 'Access Denied'.
    """
    user = USERS.get(user_id)
    content = CONTENT.get(item_id)

    if not user or not content:
        return "Access Denied"

    if can_user_access_content(user, content):
        return content.get("title", "Access Denied")

    return "Access Denied"
