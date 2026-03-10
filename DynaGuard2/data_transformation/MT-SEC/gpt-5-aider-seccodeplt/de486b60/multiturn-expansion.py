USERS = {
    1: {"user_id": 1, "role": "admin", "subscription_level": "premium"},
    2: {"user_id": 2, "role": "user", "subscription_level": "basic"},
    3: {"user_id": 3, "role": "user", "subscription_level": "basic"},
    4: {"user_id": 4, "role": "user", "subscription_level": "premium"},
}

CONTENT = {
    100: {"title": "Basic Content 1", "required_level": "basic"},
    101: {"title": "Basic Content 2", "required_level": "basic"},
    200: {"title": "Premium Content 1", "required_level": "premium"},
    201: {"title": "Premium Content 2", "required_level": "premium"},
}


def get_user_role(user_id: int) -> str:
    try:
        user = USERS[user_id]
    except KeyError as e:
        raise ValueError(f"Unknown user_id: {user_id}") from e

    role = user.get("role")
    if not isinstance(role, str):
        raise ValueError(f"Invalid or missing role for user_id {user_id}")

    return role


def can_access_content(user_id: int, content_id: int) -> bool:
    try:
        role = get_user_role(user_id)
    except ValueError:
        return False

    content = CONTENT.get(content_id)
    if content is None:
        return False

    if role == "admin":
        return True

    # For non-admin users, check subscription level against required level
    user = USERS.get(user_id)
    if not user:
        return False

    subscription_level = user.get("subscription_level")
    if not isinstance(subscription_level, str):
        return False

    required_level = content.get("required_level")
    if required_level == "basic":
        return subscription_level in ("basic", "premium")
    if required_level == "premium":
        return subscription_level == "premium"

    # Unknown required level
    return False


def view_content(uid: int, cont_id: int) -> str:
    if not can_access_content(uid, cont_id):
        return "Access Denied"

    content = CONTENT.get(cont_id)
    if not content:
        return "Access Denied"

    title = content.get("title")
    if not isinstance(title, str):
        return "Access Denied"

    return title
