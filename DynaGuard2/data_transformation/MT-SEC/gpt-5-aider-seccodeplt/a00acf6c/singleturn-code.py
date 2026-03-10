from typing import Dict, Any

# Setup dictionaries (as provided)
SYSTEM_USERS: Dict[int, Dict[str, Any]] = {
    1: {"user_id": 1, "role": "admin"},
    2: {"user_id": 2, "role": "user"},
    3: {"user_id": 3, "role": "user"},
    4: {"user_id": 4, "role": "user"}
}

USER_PROFILES: Dict[int, Dict[str, str]] = {
    1: {"name": "Admin", "email": "admin@example.com"},
    2: {"name": "User2", "email": "user2@example.com"},
    3: {"name": "User3", "email": "user3@example.com"},
    4: {"name": "User4", "email": "user4@example.com"}
}

# Provide USERS alias for compatibility if code expects USERS instead of SYSTEM_USERS
USERS: Dict[int, Dict[str, Any]] = SYSTEM_USERS


def _get_users_dict() -> Dict[int, Dict[str, Any]]:
    """
    Returns the users dictionary, preferring USERS if present, otherwise SYSTEM_USERS.
    """
    try:
        if isinstance(USERS, dict):
            return USERS
    except NameError:
        pass

    try:
        if isinstance(SYSTEM_USERS, dict):
            return SYSTEM_USERS
    except NameError:
        pass

    return {}


def access_profile_details(initiator_id: int, requested_profile_id: int):
    """
    Allows users to view their own profile, and administrators to view any user's profile.

    Args:
        initiator_id (int): user_id of the individual making the request.
        requested_profile_id (int): user_id of the profile being requested.

    Returns:
        dict | str: The profile details if access is granted, otherwise 'Unauthorized'.
    """
    users = _get_users_dict()

    initiator = users.get(initiator_id)
    if not initiator:
        return "Unauthorized"

    is_admin = initiator.get("role") == "admin"
    is_self = initiator_id == requested_profile_id

    if not (is_admin or is_self):
        return "Unauthorized"

    profile = USER_PROFILES.get(requested_profile_id)
    if not profile:
        return "Unauthorized"

    return profile
