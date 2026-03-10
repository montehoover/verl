import logging
from typing import Tuple

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


def _configure_logging():
    """
    Configure a default logger for this module if none is present.
    Uses a simple stream handler with a clear, timestamped format.
    """
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        # Prevent double logging if root logger has handlers
        logger.propagate = False


_configure_logging()


def is_admin_user(user: dict) -> bool:
    """
    Pure function: determines if a given user dict represents an admin.
    """
    return user.get("role") == "admin"


def is_self_request(active_user_id: int, profile_user_id: int) -> bool:
    """
    Pure function: checks if the requester is asking for their own profile.
    """
    return active_user_id == profile_user_id


def can_access_profile(active_user_id: int, profile_user_id: int, users: dict) -> bool:
    """
    Pure function: evaluates if the active user can access the target profile.
    Access is allowed if:
      - The active user exists and is an admin, or
      - The active user is requesting their own profile.
    """
    active_user = users.get(active_user_id)
    if not active_user:
        return False
    return is_admin_user(active_user) or is_self_request(active_user_id, profile_user_id)


def evaluate_access(active_user_id: int, profile_user_id: int, users: dict) -> Tuple[bool, str]:
    """
    Pure function: evaluates access and returns a tuple of (allowed, reason).
    Reasons:
      - "authorized": access granted
      - "active_user_not_found": requester does not exist
      - "forbidden": requester is neither admin nor the profile owner
    """
    active_user = users.get(active_user_id)
    if not active_user:
        return False, "active_user_not_found"
    if is_admin_user(active_user) or is_self_request(active_user_id, profile_user_id):
        return True, "authorized"
    return False, "forbidden"


def get_user_profile(profile_user_id: int, profiles: dict):
    """
    Pure function: fetches the profile for the given user ID from the provided profiles mapping.
    Returns None if not found.
    """
    return profiles.get(profile_user_id)


def display_user_profile(active_user_id: int, profile_user_id: int):
    """
    Returns the profile of the specified user if access is allowed.
    Access is allowed if:
      - The active user is requesting their own profile, or
      - The active user has the 'admin' role.

    Otherwise returns the string 'Unauthorized'.
    """
    allowed, reason = evaluate_access(active_user_id, profile_user_id, USERS)
    if not allowed:
        logger.info(
            "Profile access: outcome=UNAUTHORIZED reason=%s active_user_id=%s profile_user_id=%s",
            reason,
            active_user_id,
            profile_user_id,
        )
        return "Unauthorized"

    profile = get_user_profile(profile_user_id, USER_PROFILES)
    if profile is None:
        logger.info(
            "Profile access: outcome=UNAUTHORIZED reason=profile_not_found active_user_id=%s profile_user_id=%s",
            active_user_id,
            profile_user_id,
        )
        return "Unauthorized"

    logger.info(
        "Profile access: outcome=AUTHORIZED active_user_id=%s profile_user_id=%s",
        active_user_id,
        profile_user_id,
    )
    return profile
