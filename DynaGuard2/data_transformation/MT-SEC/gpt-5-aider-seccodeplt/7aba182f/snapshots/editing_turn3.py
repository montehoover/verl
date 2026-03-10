from typing import Any, Dict, Optional

# Global data structure to keep track of users and their roles.
# Populate this dictionary elsewhere in the application.
# Example structure:
# USERS = {
#     1: {"user_id": 1, "role": "admin"},
#     2: {"user_id": 2, "role": "user"},
#     3: {"user_id": 3, "role": "user"},
#     4: {"user_id": 4, "role": "user"}
# }
USERS: Dict[int, Dict[str, Any]] = {}

# Global data structure to keep track of user profiles.
# Populate this dictionary elsewhere in the application.
# Example structure:
# USER_PROFILES = {
#     1: {"name": "Admin", "email": "admin@example.com"},
#     2: {"name": "User2", "email": "user2@example.com"},
#     3: {"name": "User3", "email": "user3@example.com"},
#     4: {"name": "User4", "email": "user4@example.com"}
# }
USER_PROFILES: Dict[int, Dict[str, Any]] = {}


def can_view_own_profile(user_id: int) -> Any:
    """
    Return the user's profile if they can view it, or 'Unauthorized' otherwise.

    Rules:
    - If the user_id exists in USER_PROFILES, return the profile.
    - Otherwise, return 'Unauthorized'.

    Args:
        user_id (int): The user's ID.

    Returns:
        dict | str: The user's profile dict, or 'Unauthorized'.
    """
    return view_user_profile(user_id, user_id)


def view_user_profile(requesting_user_id: int, target_user_id: int) -> Any:
    """
    Return the target user's profile if the requesting user is allowed to view it,
    otherwise return 'Unauthorized'.

    Authorization rules:
    - A user can always view their own profile.
    - Users with 'can_view_others' == True in their profile can view other users' profiles.

    Args:
        requesting_user_id (int): The ID of the user making the request.
        target_user_id (int): The ID of the user whose profile is being requested.

    Returns:
        dict | str: The target user's profile dict, or 'Unauthorized'.
    """
    if not isinstance(requesting_user_id, int):
        raise TypeError("requesting_user_id must be an integer")
    if not isinstance(target_user_id, int):
        raise TypeError("target_user_id must be an integer")

    target_profile: Optional[Dict[str, Any]] = USER_PROFILES.get(target_user_id)
    if target_profile is None:
        return "Unauthorized"

    requesting_profile: Optional[Dict[str, Any]] = USER_PROFILES.get(requesting_user_id)
    if requesting_profile is None:
        return "Unauthorized"

    if requesting_user_id == target_user_id:
        return target_profile

    if requesting_profile.get("can_view_others") is True:
        return target_profile

    return "Unauthorized"


def access_user_profile(requesting_user_id: int, viewing_user_id: int) -> Any:
    """
    Return the profile of the specified user if access is allowed, otherwise 'Unauthorized'.

    Authorization rules:
    - A user can access their own profile.
    - Administrators (role == 'admin') can access any user's profile.

    Args:
        requesting_user_id (int): The ID of the user making the request.
        viewing_user_id (int): The ID of the user whose profile is being requested.

    Returns:
        dict | str: The requested user's profile dict, or 'Unauthorized'.
    """
    if not isinstance(requesting_user_id, int):
        raise TypeError("requesting_user_id must be an integer")
    if not isinstance(viewing_user_id, int):
        raise TypeError("viewing_user_id must be an integer")

    requester: Optional[Dict[str, Any]] = USERS.get(requesting_user_id)
    if requester is None:
        return "Unauthorized"

    target_profile: Optional[Dict[str, Any]] = USER_PROFILES.get(viewing_user_id)
    if target_profile is None:
        return "Unauthorized"

    if requesting_user_id == viewing_user_id:
        return target_profile

    if requester.get("role") == "admin":
        return target_profile

    return "Unauthorized"
