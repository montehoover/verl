user_data_store = []

def store_user_info(user_id: str, user_info: str) -> bool:
    """
    Stores user information into a global list.

    Args:
        user_id: The ID of the user.
        user_info: Information about the user.

    Returns:
        True if the information was successfully stored, False otherwise.
    """
    try:
        user_data_store.append({'user_id': user_id, 'user_info': user_info})
        return True
    except Exception:
        # In a simple list append, an exception is highly unlikely 
        # unless there are memory issues, but we'll keep this for robustness.
        return False
