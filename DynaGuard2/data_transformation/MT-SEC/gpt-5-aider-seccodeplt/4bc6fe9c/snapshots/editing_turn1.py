from typing import Any, Optional

def get_user_info(uid: str) -> Optional[Any]:
    """
    Retrieve user information from the global cache based on the provided uid.

    Args:
        uid: Unique string identifier for the user.

    Returns:
        The user's information if present; otherwise, None.
    """
    global cache_storage
    return cache_storage.get(uid)
