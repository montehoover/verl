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

def grant_access(user_id: int, item_id: int):
    """
    Manages content access based on the user's role and subscription status.

    Args:
        user_id: int, representing the user trying to access the content.
        item_id: int, representing the intended content.

    Returns:
        The content title if the user is permitted to access it; otherwise, 'Access Denied'.
    """
    if user_id not in USERS or item_id not in CONTENT:
        return 'Access Denied'

    user = USERS[user_id]
    content_item = CONTENT[item_id]

    if user["role"] == "admin":
        return content_item["title"]

    if user["role"] == "user":
        user_level = user["subscription_level"]
        required_level = content_item["required_level"]

        if required_level == "basic":
            # Basic and Premium users can access basic content
            return content_item["title"]
        elif required_level == "premium":
            if user_level == "premium":
                return content_item["title"]
            else:
                return 'Access Denied'
        else:
            # Unknown required_level
            return 'Access Denied'
            
    return 'Access Denied'
