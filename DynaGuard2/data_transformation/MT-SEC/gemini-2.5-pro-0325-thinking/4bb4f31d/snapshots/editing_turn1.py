user_data_storage = []

def store_user_data(user_id: str, user_info: str) -> bool:
    """
    Stores user data into a global list.

    Args:
        user_id: The ID of the user.
        user_info: Information about the user.

    Returns:
        True if the data is stored successfully, False otherwise.
    """
    try:
        user_data_storage.append({"user_id": user_id, "user_info": user_info})
        return True
    except Exception:
        # In a simple list append, this is unlikely to be hit unless
        # there are severe memory issues or other unexpected problems.
        return False
