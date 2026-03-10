USER_INFO_LIST = []  # Global list to store (user_id, user_info) tuples

def store_user_info(user_id, user_info):
    """
    Store the given user_id and user_info in the global list.
    Returns True if the operation is successful, otherwise False.
    """
    if not isinstance(user_id, str) or not isinstance(user_info, str):
        return False
    try:
        USER_INFO_LIST.append((user_id, user_info))
        return True
    except Exception:
        return False
