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


def view_profile_with_access(active_user_id: int, profile_user_id: int):
    """
    Allows admins to view any user's profile; regular users can only view their own.

    Args:
        active_user_id: ID of the user making the request.
        profile_user_id: ID of the profile being requested.

    Returns:
        The requested user's profile (dict or None if not found) when access is allowed,
        otherwise the string 'Unauthorized'.

    Raises:
        TypeError: If either argument is not an integer.
    """
    if not isinstance(active_user_id, int) or not isinstance(profile_user_id, int):
        raise TypeError("active_user_id and profile_user_id must be integers")

    active_profile = get_user_profile(active_user_id)
    if active_profile is None:
        return "Unauthorized"

    role = str(active_profile.get("role", "")).lower()
    if role == "admin" or active_user_id == profile_user_id:
        return get_user_profile(profile_user_id)

    return "Unauthorized"
