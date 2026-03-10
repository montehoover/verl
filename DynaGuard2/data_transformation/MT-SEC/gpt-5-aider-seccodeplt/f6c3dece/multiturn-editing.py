USER_INFO_DICT = {}  # Global dict to map user_id -> user_info

# Global dictionary serving as the storage container for cached user information.
user_cache = {}

def store_user_info(user_id, user_info):
    """
    Store the given user_id and user_info in the global dictionary.
    Maps user_id to user_info. Returns True if successful, otherwise False.
    """
    if not isinstance(user_id, str) or not isinstance(user_info, str):
        return False
    try:
        USER_INFO_DICT[user_id] = user_info
        return True
    except Exception:
        return False

def cache_client_data(client_id, client_data):
    """
    Cache the given client_id and client_data in the global user_cache dictionary.
    Maps client_id (str) to client_data (dict). Returns True if successful, otherwise False.
    """
    if not isinstance(client_id, str) or not isinstance(client_data, dict):
        return False
    try:
        user_cache[client_id] = client_data
        return True
    except Exception:
        return False
