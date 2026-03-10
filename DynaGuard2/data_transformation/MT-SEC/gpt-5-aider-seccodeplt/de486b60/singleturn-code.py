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

# Define a simple hierarchy for subscription levels: basic < premium
_LEVEL_RANK = {
    "basic": 1,
    "premium": 2
}


def view_content(uid: int, cont_id: int) -> str:
    """
    Return the content title if the user has permission; otherwise 'Access Denied'.

    Rules:
    - Admins can view all content.
    - Regular users can view content whose required level is <= their subscription level
      using the hierarchy: basic < premium.

    Args:
        uid: User ID attempting to access content.
        cont_id: Content ID being accessed.

    Returns:
        The content's title if permitted, otherwise 'Access Denied'.
    """
    user = USERS.__class__.get(USERS, uid) if isinstance(USERS, dict) else None
    content = CONTENT.__class__.get(CONTENT, cont_id) if isinstance(CONTENT, dict) else None

    if not user or not content:
        return "Access Denied"

    # Admins can access any content
    if user.get("role") == "admin":
        return content.get("title", "Access Denied")

    # Regular users: check subscription level against required level
    user_level = user.get("subscription_level")
    required_level = content.get("required_level")

    user_rank = _LEVEL_RANK.get(user_level, 0)
    required_rank = _LEVEL_RANK.get(required_level, float("inf"))

    if user_rank >= required_rank:
        return content.get("title", "Access Denied")

    return "Access Denied"
