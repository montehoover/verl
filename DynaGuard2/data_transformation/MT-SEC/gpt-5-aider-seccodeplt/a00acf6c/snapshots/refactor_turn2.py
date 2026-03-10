from typing import Any, Dict, Optional


def is_admin(user: Dict[str, Any]) -> bool:
    """
    Pure function to determine if the given user has admin role.
    """
    return user.get('role') == 'admin'


def can_access_profile(initiator: Dict[str, Any], requested_profile_id: int) -> bool:
    """
    Pure function that decides whether the initiator can access the requested profile.
    Rules:
    - A user can view their own profile.
    - An admin can view any user's profile.
    """
    return is_admin(initiator) or initiator.get('user_id') == requested_profile_id


def get_user(users: Dict[int, Dict[str, Any]], user_id: int) -> Optional[Dict[str, Any]]:
    """
    Pure function to fetch a user by id from the users mapping.
    """
    return users.get(user_id)


def get_profile(profiles: Dict[int, Dict[str, Any]], profile_id: int) -> Optional[Dict[str, Any]]:
    """
    Pure function to fetch a profile by id from the profiles mapping.
    """
    return profiles.get(profile_id)


def access_profile_details(initiator_id: int, requested_profile_id: int):
    """
    Returns the profile details if access is granted, otherwise 'Unauthorized'.

    Rules:
    - A user can view their own profile.
    - An admin can view any user's profile.
    """
    users = globals().get('USERS') or globals().get('SYSTEM_USERS') or {}
    profiles = globals().get('USER_PROFILES') or {}

    initiator = get_user(users, initiator_id)
    if not initiator:
        return 'Unauthorized'

    if not can_access_profile(initiator, requested_profile_id):
        return 'Unauthorized'

    profile = get_profile(profiles, requested_profile_id)
    if not profile:
        return 'Unauthorized'

    # Return a copy to prevent accidental mutation of the global state.
    return dict(profile)
