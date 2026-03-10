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
    "basic": 0,
    "premium": 1
}


def can_access_level(user_id, required_level):
    """
    Determine whether a user is authorized to access content based on their
    subscription level and role.

    Args:
        user_id (int): The ID of the user to check.
        required_level (str): The required subscription level for the content.
                              Supported levels: "basic", "premium".

    Returns:
        bool: True if access should be granted, False otherwise.
    """
    user = USERS.get(user_id)
    if not user:
        return False

    role = (user.get("role") or "").lower()
    if role == "admin":
        return True

    # Validate required level
    if not isinstance(required_level, str):
        return False
    required_level_norm = required_level.lower()
    if required_level_norm not in LEVEL_ORDER:
        return False

    # Validate user's subscription level
    user_level = (user.get("subscription_level") or "").lower()
    if user_level not in LEVEL_ORDER:
        return False

    return LEVEL_ORDER[user_level] >= LEVEL_ORDER[required_level_norm]
