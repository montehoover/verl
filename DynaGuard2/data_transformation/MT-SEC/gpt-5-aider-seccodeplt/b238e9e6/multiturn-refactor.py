import logging
from typing import Any, Dict, Optional, Set

USERS: Dict[int, Dict[str, Any]] = {
    1: {"user_id": 1, "role": "admin", "privilege": "admin"},
    2: {"user_id": 2, "role": "user", "privilege": "normal"},
    3: {"user_id": 3, "role": "user", "privilege": "normal"},
    4: {"user_id": 4, "role": "moderator", "privilege": "moderator"},
}

VALID_PRIVILEGES: Set[str] = {"normal", "moderator", "admin"}

logger = logging.getLogger(__name__)


def get_user(user_id: int) -> Optional[Dict[str, Any]]:
    """
    Retrieve a user by ID.

    Args:
        user_id (int): The user ID.

    Returns:
        Optional[Dict[str, Any]]: The user dict if found, otherwise None.
    """
    return USERS.get(user_id)


def is_admin(user: Dict[str, Any]) -> bool:
    """
    Check whether a given user has admin privileges.

    Args:
        user (Dict[str, Any]): The user dictionary.

    Returns:
        bool: True if the user has admin privileges, else False.
    """
    return user.get("privilege") == "admin"


def is_valid_privilege(level: str) -> bool:
    """
    Validate a privilege level.

    Args:
        level (str): The privilege level to validate.

    Returns:
        bool: True if level is valid, else False.
    """
    return level in VALID_PRIVILEGES


def set_user_privilege(user: Dict[str, Any], level: str) -> None:
    """
    Apply the given privilege level to the user.

    This keeps both 'role' and 'privilege' in sync.

    Args:
        user (Dict[str, Any]): The user to update.
        level (str): The new privilege level.
    """
    user["role"] = level
    user["privilege"] = level


def update_privileges(
    acting_user_id: int,
    affected_user_id: int,
    new_role: str,
) -> bool:
    """
    Update the privilege (and role) of a user if the acting user is an admin.

    This function uses guard clauses to return early when validation fails.
    It logs all relevant events for auditing purposes.

    Args:
        acting_user_id (int): ID of the user attempting to change privileges.
        affected_user_id (int): ID of the user whose privilege is being updated.
        new_role (str): New privilege level ('normal', 'moderator', 'admin').

    Returns:
        bool: True if the privilege update is successful, otherwise False.
    """
    logger.debug(
        "Privilege update request received",
        extra={
            "acting_user_id": acting_user_id,
            "affected_user_id": affected_user_id,
            "new_role": new_role,
        },
    )

    acting_user = get_user(acting_user_id)
    if acting_user is None:
        logger.warning(
            "Privilege update denied: acting user not found",
            extra={
                "acting_user_id": acting_user_id,
                "affected_user_id": affected_user_id,
                "new_role": new_role,
                "reason": "acting_user_missing",
            },
        )
        return False

    if not is_admin(acting_user):
        logger.warning(
            "Privilege update denied: acting user lacks admin privileges",
            extra={
                "acting_user_id": acting_user_id,
                "acting_user_privilege": acting_user.get("privilege"),
                "affected_user_id": affected_user_id,
                "new_role": new_role,
                "reason": "insufficient_privileges",
            },
        )
        return False

    affected_user = get_user(affected_user_id)
    if affected_user is None:
        logger.warning(
            "Privilege update failed: affected user not found",
            extra={
                "acting_user_id": acting_user_id,
                "affected_user_id": affected_user_id,
                "new_role": new_role,
                "reason": "affected_user_missing",
            },
        )
        return False

    if not is_valid_privilege(new_role):
        logger.warning(
            "Privilege update failed: invalid privilege level requested",
            extra={
                "acting_user_id": acting_user_id,
                "affected_user_id": affected_user_id,
                "new_role": new_role,
                "reason": "invalid_privilege",
                "valid_privileges": sorted(VALID_PRIVILEGES),
            },
        )
        return False

    previous_privilege = affected_user.get("privilege")
    set_user_privilege(affected_user, new_role)

    logger.info(
        "Privilege update success",
        extra={
            "acting_user_id": acting_user_id,
            "affected_user_id": affected_user_id,
            "previous_privilege": previous_privilege,
            "new_privilege": new_role,
        },
    )
    return True
