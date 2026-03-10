import logging

# Provide a fallback USERS dictionary for isolated lint/test environments.
# In production, this can be overridden by defining USERS before importing this module.
if 'USERS' not in globals():
    USERS = {
        1: {"user_id": 1, "role": "admin", "privilege": "admin"},
        2: {"user_id": 2, "role": "user", "privilege": "normal"},
        3: {"user_id": 3, "role": "user", "privilege": "normal"},
        4: {"user_id": 4, "role": "moderator", "privilege": "moderator"},
    }

ALLOWED_PRIVILEGES = frozenset({"normal", "moderator", "admin"})

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def is_valid_privilege(new_privilege: str) -> bool:
    """
    Pure function: Validate the provided privilege value.

    Args:
        new_privilege (str): The proposed privilege value.

    Returns:
        bool: True if valid, False otherwise.
    """
    return isinstance(new_privilege, str) and new_privilege in ALLOWED_PRIVILEGES


def is_admin(user: dict) -> bool:
    """
    Pure function: Determine if the provided user has admin privileges.

    Args:
        user (dict): A user dictionary.

    Returns:
        bool: True if the user has admin privilege, False otherwise.
    """
    if not isinstance(user, dict):
        return False
    return user.get("privilege") == "admin"


def updated_user_with_privilege(user: dict, new_privilege: str) -> dict:
    """
    Pure function: Return a new user dict with the updated privilege applied.

    Args:
        user (dict): The original user dictionary.
        new_privilege (str): The new privilege value.

    Returns:
        dict: A new user dictionary with the updated privilege.
    """
    return {**user, "privilege": new_privilege}


def _log_privilege_change_attempt(
    current_user_id: int,
    target_user_id: int,
    new_privilege: str,
    success: bool,
    reason: str | None = None,
) -> None:
    """
    Log a privilege modification attempt with outcome.

    This function is intentionally not pure (it performs I/O via logging).
    """
    if reason:
        logger.info(
            "privilege_change_attempt current_user_id=%s target_user_id=%s new_privilege=%s success=%s reason=%s",
            current_user_id,
            target_user_id,
            new_privilege,
            success,
            reason,
        )
    else:
        logger.info(
            "privilege_change_attempt current_user_id=%s target_user_id=%s new_privilege=%s success=%s",
            current_user_id,
            target_user_id,
            new_privilege,
            success,
        )


def modify_user_privileges(current_user_id: int, target_user_id: int, new_privilege: str) -> bool:
    """
    Modify the privilege level of a target user, only if the current user is an admin.

    Args:
        current_user_id (int): The user_id of the actor attempting to modify privileges.
        target_user_id (int): The user_id of the user whose privileges are being modified.
        new_privilege (str): The new privilege level to assign ('normal', 'moderator', or 'admin').

    Returns:
        bool: True if the privilege was successfully modified, else False.
    """
    # Validate that both user IDs are integers.
    if not isinstance(current_user_id, int) or not isinstance(target_user_id, int):
        _log_privilege_change_attempt(current_user_id, target_user_id, new_privilege, False, "invalid_user_id_type")
        return False

    # Validate the new privilege level via a pure function.
    if not is_valid_privilege(new_privilege):
        _log_privilege_change_attempt(current_user_id, target_user_id, new_privilege, False, "invalid_privilege")
        return False

    # Fetch users safely.
    current_user = USERS.get(current_user_id)
    target_user = USERS.get(target_user_id)

    # Ensure both users exist.
    if current_user is None:
        _log_privilege_change_attempt(current_user_id, target_user_id, new_privilege, False, "current_user_not_found")
        return False

    if target_user is None:
        _log_privilege_change_attempt(current_user_id, target_user_id, new_privilege, False, "target_user_not_found")
        return False

    # Only admins can modify privileges.
    if not is_admin(current_user):
        _log_privilege_change_attempt(current_user_id, target_user_id, new_privilege, False, "not_admin")
        return False

    # Apply the privilege change using a pure transformer.
    USERS[target_user_id] = updated_user_with_privilege(target_user, new_privilege)
    _log_privilege_change_attempt(current_user_id, target_user_id, new_privilege, True)
    return True
