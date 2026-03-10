from typing import List, Tuple

# Global list to store user data as tuples of (user_id, user_info)
USER_DATA: List[Tuple[str, str]] = []


def store_user_data(user_id: str, user_info: str) -> bool:
    """
    Store user data in the global USER_DATA list.

    Args:
        user_id: The user's identifier as a string.
        user_info: The user's information as a string.

    Returns:
        True if data is stored successfully, False otherwise.
    """
    try:
        if not isinstance(user_id, str) or not isinstance(user_info, str):
            return False
        USER_DATA.append((user_id, user_info))
        return True
    except Exception:
        return False
