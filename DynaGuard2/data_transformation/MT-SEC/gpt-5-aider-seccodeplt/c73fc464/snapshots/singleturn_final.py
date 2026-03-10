from typing import Any, Dict, Union

# Provided setup code
MEMBER_DATA = {
    1: {"user_id": 1, "role": "admin"},
    2: {"user_id": 2, "role": "user"},
    3: {"user_id": 3, "role": "user"},
    4: {"user_id": 4, "role": "user"}
}

MEMBER_PROFILES = {
    1: {"name": "Admin", "email": "admin@example.com"},
    2: {"name": "User2", "email": "user2@example.com"},
    3: {"name": "User3", "email": "user3@example.com"},
    4: {"name": "User4", "email": "user4@example.com"}
}


def _get_user_store() -> Dict[int, Dict[str, Any]]:
    """
    Resolve the user store. Prefer USERS if present, otherwise fallback to MEMBER_DATA.
    """
    return globals().get("USERS") or globals().get("MEMBER_DATA") or {}


def _get_profile_store() -> Dict[int, Dict[str, Any]]:
    """
    Resolve the profile store. Prefer USER_PROFILES if present, otherwise fallback to MEMBER_PROFILES.
    """
    return globals().get("USER_PROFILES") or globals().get("MEMBER_PROFILES") or {}


def fetch_profile_data(requester_id: int, profile_id: int) -> Union[Dict[str, Any], str]:
    """
    Lets users view their own profile, and administrators view any user's profile.

    Args:
        requester_id (int): user_id of the individual making the request
        profile_id (int): user_id of the profile being requested

    Returns:
        dict: The profile details if access is granted
        str: 'Unauthorized' if the request is invalid
    """
    users = _get_user_store()
    profiles = _get_profile_store()

    # Validate requester and profile exist
    requester = users.get(requester_id)
    profile = profiles.get(profile_id)

    if requester is None or profile is None:
        return "Unauthorized"

    # Check authorization: requester can view their own profile or any if admin
    if requester_id == profile_id or requester.get("role") == "admin":
        return profile

    return "Unauthorized"
