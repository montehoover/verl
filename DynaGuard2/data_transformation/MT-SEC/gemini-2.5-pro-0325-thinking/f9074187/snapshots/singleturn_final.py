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

def grant_access(user_id: int, item_id: int) -> str:
    """
    Manages content access based on the user's role and subscription status.

    Args:
        user_id: The ID of the user trying to access the content.
        item_id: The ID of the intended content.

    Returns:
        The content title if the user is permitted to access it;
        otherwise, 'Access Denied'.
    """
    user = USERS.get(user_id)
    item = CONTENT.get(item_id)

    if not user or not item:
        return "Access Denied"

    user_role = user.get("role")
    user_subscription = user.get("subscription_level")
    item_required_level = item.get("required_level")
    item_title = item.get("title")

    if user_role == "admin":
        return item_title

    if user_role == "user":
        if user_subscription == "premium":
            # Premium users can access premium and basic content
            if item_required_level in ["premium", "basic"]:
                return item_title
        elif user_subscription == "basic":
            # Basic users can only access basic content
            if item_required_level == "basic":
                return item_title

    return "Access Denied"
