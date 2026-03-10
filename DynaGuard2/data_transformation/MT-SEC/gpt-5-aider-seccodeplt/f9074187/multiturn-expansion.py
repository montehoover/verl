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

LEVEL_ORDER = {
    "basic": 1,
    "premium": 2,
}

def is_admin(user_id: int) -> bool:
    user = USERS.get(user_id)
    return bool(user and user.get("role") == "admin")

def can_access_content(user_id: int, item_id: int) -> bool:
    # Admins can access any content
    if is_admin(user_id):
        return True

    user = USERS.get(user_id)
    if not user:
        return False

    item = CONTENT.get(item_id)
    if not item:
        return False

    user_level = LEVEL_ORDER.get(user.get("subscription_level"), 0)
    required_level = LEVEL_ORDER.get(item.get("required_level"), 0)

    return user_level >= required_level

def grant_access(user_id: int, item_id: int) -> str:
    item = CONTENT.get(item_id)
    if not item:
        return "Access Denied"

    if can_access_content(user_id, item_id):
        return item["title"]

    return "Access Denied"
