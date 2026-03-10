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


def view_content(current_user_id: int, content_id: int) -> str:
    """
    Return the title of the content if the current user is an admin.
    Otherwise, return "Access Denied" regardless of subscription level.
    """
    user = USERS.get(current_user_id)
    if not user:
        return "Access Denied"

    if user.get("role") == "admin":
        content = CONTENT.get(content_id)
        if content is None:
            return "Content Not Found"
        return content.get("title", "Content Not Found")

    return "Access Denied"
