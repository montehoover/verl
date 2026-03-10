cache_storage = {}

def store_user_data(uid: str, data: dict) -> bool:
    """
    Stores new user information in the global cache.

    Args:
        uid: The unique string identifier for the user.
        data: A dictionary containing user details.

    Returns:
        True if the data was successfully stored, False otherwise.
    """
    try:
        cache_storage[uid] = data
        return True
    except Exception:
        # In a real scenario, log the exception
        return False
