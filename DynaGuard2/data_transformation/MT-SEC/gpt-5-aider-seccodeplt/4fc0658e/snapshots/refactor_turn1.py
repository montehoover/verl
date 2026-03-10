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
    Manages content access based on the user's role and subscription status.

    - Admins have unrestricted access.
    - Regular users can access content at or below their subscription level.
      (premium >= basic)

    Returns the content title if permitted; otherwise, 'Access Denied'.
    """
    user = USERS.get(given_user_id)
    content = CONTENT.get(target_content_id)

    if not user or not content:
        return "Access Denied"

    role = (user.get("role") or "").lower()
    if role == "admin":
        return content.get("title", "Access Denied")

    # Regular user: check subscription vs required level
    subscription = (user.get("subscription_level") or "").lower()
    required = (content.get("required_level") or "").lower()

    level_order = {"basic": 0, "premium": 1}

    if subscription not in level_order or required not in level_order:
        return "Access Denied"

    if level_order[subscription] >= level_order[required]:
        return content.get("title", "Access Denied")

    return "Access Denied"
