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

    Admins have unrestricted access, while regular users are only allowed to
    access content that aligns with their subscription level.

    Args:
        given_user_id: The ID of the user trying to access the content.
        target_content_id: The ID of the intended content.

    Returns:
        The content title if the user is permitted to access it;
        otherwise, 'Access Denied'.
    """
    user = USERS.get(given_user_id)
    content = CONTENT.get(target_content_id)

    if not user or not content:
        return "Access Denied"

    if user["role"] == "admin":
        return content["title"]

    if user["role"] == "user":
        if content["required_level"] == "basic":
            return content["title"]
        if content["required_level"] == "premium" and user["subscription_level"] == "premium":
            return content["title"]

    return "Access Denied"
