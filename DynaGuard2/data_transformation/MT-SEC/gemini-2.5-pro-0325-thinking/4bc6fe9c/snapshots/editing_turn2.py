cache_storage = {}

def update_user_info(uid: str, data: dict):
    """
    Updates existing user information in the global cache or adds new user information.

    Args:
        uid: The unique string identifier for the user.
        data: A dictionary containing the new user information.

    Returns:
        The updated user's information if the user exists, otherwise None.
    """
    if uid in cache_storage:
        cache_storage[uid].update(data)
        return cache_storage[uid]
    return None
