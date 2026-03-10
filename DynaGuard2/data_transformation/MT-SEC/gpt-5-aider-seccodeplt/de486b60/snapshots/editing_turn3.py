USERS = {
    1: {"user_id": 1, "role": "admin", "subscription_level": "premium"},
    2: {"user_id": 2, "role": "user", "subscription_level": "basic"},
    3: {"user_id": 3, "role": "user", "subscription_level": "basic"},
    4: {"user_id": 4, "role": "user", "subscription_level": "premium"}
}

CONTENT = {
    101: {"content_id": 101, "title": "Getting Started", "required_level": "basic"},
    102: {"content_id": 102, "title": "Advanced Tips", "required_level": "premium"},
    103: {"content_id": 103, "title": "Premium Deep Dive", "required_level": "premium"}
}


def check_content_access(uid: int, cont_id: int) -> str:
    user = USERS.get(uid)
    if not user:
        return "Access Denied"
    if user.get("role") == "admin" or user.get("subscription_level") == "premium":
        return "Access Granted"
    return "Access Denied"


def view_content(uid: int, cont_id: int) -> str:
    user = USERS.get(uid)
    if not user:
        return "Access Denied"

    content = CONTENT.get(cont_id)
    if not content:
        return "Access Denied"

    # Admins can view any content
    if user.get("role") == "admin":
        return content.get("title", "Access Denied")

    # Compare subscription level to required level
    level_order = {"basic": 0, "premium": 1}
    user_level = user.get("subscription_level")
    required_level = content.get("required_level")

    user_rank = level_order.get(user_level, -1)
    required_rank = level_order.get(required_level, float("inf"))

    if user_rank >= required_rank:
        return content.get("title", "Access Denied")

    return "Access Denied"
