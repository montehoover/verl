user_data_storage = {}

def store_user_data(user_id: str, user_info: dict) -> bool:
    """
    Stores user data into a global dictionary.

    Args:
        user_id: The ID of the user.
        user_info: A dictionary containing user details (e.g., 'name', 'email').

    Returns:
        True if the data is stored successfully, False otherwise.
    """
    try:
        user_data_storage[user_id] = user_info
        return True
    except Exception:
        # This is unlikely to be hit for a simple dictionary assignment
        # unless there are severe memory issues or other unexpected problems.
        return False

user_cache = {}

def cache_application_data(client_id: str, info_data: dict) -> bool:
    """
    Caches user-related information to optimize application speed.

    Args:
        client_id: The unique ID of the client (user).
        info_data: A dictionary containing user-related information.

    Returns:
        True if the data is cached successfully, False otherwise.
    """
    try:
        user_cache[client_id] = info_data
        return True
    except Exception:
        # This is unlikely to be hit for a simple dictionary assignment
        # unless there are severe memory issues or other unexpected problems.
        return False
