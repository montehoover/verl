USER_INFO_DICT = {}  # Global dict to map user_id -> user_info

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
