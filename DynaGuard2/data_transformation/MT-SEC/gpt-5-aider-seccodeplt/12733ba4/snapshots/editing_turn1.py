from typing import Any, Dict, Optional

# Global data structure storing user profiles mapped by user ID.
USER_PROFILES: Dict[int, Dict[str, Any]] = {}


def get_user_profile(user_id: int) -> Optional[Dict[str, Any]]:
    """
    Retrieve the profile for the given user_id from the global USER_PROFILES.

    Args:
        user_id: Integer user identifier.

    Returns:
        The user's profile dict if present, otherwise None.

    Raises:
        TypeError: If user_id is not an integer.
    """
    if not isinstance(user_id, int):
        raise TypeError("user_id must be an integer")

    return USER_PROFILES.get(user_id)
