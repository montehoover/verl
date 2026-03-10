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


def view_content(uid: int, cont_id: int) -> str:
    user = USERS.get(uid)
    content = CONTENT.get(cont_id)

    if not user or not content:
        return "Access Denied"

    # Admins have access to all content
    if user.get("role") == "admin":
        return content["title"]

    required_level = content.get("required_level")
    user_level = user.get("subscription_level")

    # Define level hierarchy
    level_rank = {"basic": 0, "premium": 1}

    if required_level not in level_rank or user_level not in level_rank:
        return "Access Denied"

    if level_rank[user_level] >= level_rank[required_level]:
        return content["title"]

    return "Access Denied"
