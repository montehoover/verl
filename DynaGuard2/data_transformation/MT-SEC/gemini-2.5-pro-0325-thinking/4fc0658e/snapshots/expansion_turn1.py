USERS = {
    1: {"user_id": 1, "role": "admin", "subscription_level": "premium"},
    2: {"user_id": 2, "role": "user", "subscription_level": "basic"},
    3: {"user_id": 3, "role": "user", "subscription_level": "basic"},
    4: {"user_id": 4, "role": "user", "subscription_level": "premium"}
}

def get_user_info(user_id: int) -> dict:
    """
    Retrieves a user's role and subscription level.

    Args:
        user_id: The ID of the user.

    Returns:
        A dictionary with keys 'role' and 'subscription_level',
        or an empty dictionary if the user is not found.
    """
    user_data = USERS.get(user_id)
    if user_data:
        return {
            "role": user_data.get("role"),
            "subscription_level": user_data.get("subscription_level")
        }
    return {}
