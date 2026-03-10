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

_SUBSCRIPTION_RANK = {
    "basic": 1,
    "premium": 2
}

def grant_access(user_id: int, item_id: int) -> str:
    """
    Determine if a user can access a piece of content.

    Args:
        user_id: The ID of the user attempting access.
        item_id: The ID of the content item.

    Returns:
        The content title if access is permitted; otherwise, "Access Denied".
    """
    user = USERS.get(user_id)
    content = CONTENT.get(item_id)

    if not user or not content:
        return "Access Denied"

    role = str(user.get("role", "")).lower()
    if role == "admin":
        return content["title"]

    user_level = str(user.get("subscription_level", "")).lower()
    required_level = str(content.get("required_level", "")).lower()

    user_rank = _SUBSCRIPTION_RANK.get(user_level, 0)
    required_rank = _SUBSCRIPTION_RANK.get(required_level, float("inf"))

    if user_rank >= required_rank:
        return content["title"]

    return "Access Denied"
