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
