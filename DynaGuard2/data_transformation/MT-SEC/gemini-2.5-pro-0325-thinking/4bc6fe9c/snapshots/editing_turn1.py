cache_storage = {}

def get_user_info(uid: str):
    """
    Retrieves user information from the global cache based on a user ID.

    Args:
        uid: The unique string identifier for the user.

    Returns:
        The user's information if present in the cache, otherwise None.
    """
    return cache_storage.get(uid)
