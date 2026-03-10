user_data_store = {}

def store_user_info(user_id: str, user_info: str) -> bool:
    """
    Stores user information into a global dictionary.

    Args:
        user_id: The ID of the user.
        user_info: Information about the user.

    Returns:
        True if the information was successfully stored, False otherwise.
    """
    try:
        user_data_store[user_id] = user_info
        return True
    except Exception:
        # An exception is highly unlikely unless there are memory issues.
        return False

user_cache = {}

def cache_client_data(client_id: str, client_data: dict) -> bool:
    """
    Caches client data into an in-memory dictionary.

    Args:
        client_id: The ID of the client.
        client_data: A dictionary containing client data.

    Returns:
        True if the data was successfully cached, False otherwise.
    """
    try:
        user_cache[client_id] = client_data
        return True
    except Exception:
        # An exception is highly unlikely unless there are memory issues.
        return False
