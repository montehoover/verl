from typing import List, Dict

# Global list to store user information
USER_STORE: List[Dict[str, str]] = []


def store_user_info(user_id: str, user_name: str) -> bool:
    """
    Store user information in the global USER_STORE list.

    Args:
        user_id (str): The user's ID.
        user_name (str): The user's name.

    Returns:
        bool: True if the information was stored successfully, False otherwise.
    """
    if not isinstance(user_id, str) or not isinstance(user_name, str):
        return False

    try:
        USER_STORE.append({"user_id": user_id, "user_name": user_name})
        return True
    except Exception:
        return False
