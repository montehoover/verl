import logging

USERS = {
    1: {"user_id": 1, "role": "admin"},
    2: {"user_id": 2, "role": "user"},
    3: {"user_id": 3, "role": "user"},
    4: {"user_id": 4, "role": "user"}
}

USER_PROFILES = {
    1: {"name": "Admin", "email": "admin@example.com"},
    2: {"name": "User2", "email": "user2@example.com"},
    3: {"name": "User3", "email": "user3@example.com"},
    4: {"name": "User4", "email": "user4@example.com"}
}

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def is_admin(user: dict) -> bool:
    """
    Pure function: returns True if the provided user dict represents an admin.
    """
    return bool(user) and user.get("role") == "admin"


def can_view_profile(users: dict, requesting_user_id: int, viewing_user_id: int) -> bool:
    """
    Pure function: determines if the requesting user can view the viewing user's profile.
    - Admins can view any profile.
    - Users can view their own profile.
    """
    return (
        requesting_user_id in users
        and (
            is_admin(users[requesting_user_id])
            or requesting_user_id == viewing_user_id
        )
    )


def get_user_profile(user_profiles: dict, user_id: int):
    """
    Pure function: retrieves a user profile by ID, or None if not found.
    """
    return user_profiles.get(user_id)


def access_user_profile(requesting_user_id: int, viewing_user_id: int):
    """
    Grants access to a user profile based on role and ownership.

    - Admins can view any user's profile.
    - Users can view their own profile.
    Returns the profile dict if access is allowed, otherwise 'Unauthorized'.
    """
    logger.info(
        "Access request: requester_id=%s viewing_id=%s",
        requesting_user_id,
        viewing_user_id,
    )

    if not can_view_profile(USERS, requesting_user_id, viewing_user_id):
        logger.warning(
            "Access denied: requester_id=%s viewing_id=%s",
            requesting_user_id,
            viewing_user_id,
        )
        return "Unauthorized"

    profile = get_user_profile(USER_PROFILES, viewing_user_id)
    if profile is None:
        logger.warning(
            "Profile not found: viewing_id=%s (requester_id=%s)",
            viewing_user_id,
            requesting_user_id,
        )
        return "Unauthorized"

    logger.info(
        "Access granted: requester_id=%s viewing_id=%s",
        requesting_user_id,
        viewing_user_id,
    )
    return profile
