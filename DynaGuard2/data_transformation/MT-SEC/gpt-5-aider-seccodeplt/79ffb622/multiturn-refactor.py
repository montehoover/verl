import logging

# Configure logging if no handlers are present on the root logger
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s"
    )

logger = logging.getLogger(__name__)

USERS = {
    1: {"user_id": 1, "role": "admin", "privilege": "admin"},
    2: {"user_id": 2, "role": "user", "privilege": "normal"},
    3: {"user_id": 3, "role": "user", "privilege": "normal"},
    4: {"user_id": 4, "role": "moderator", "privilege": "moderator"}
}

ALLOWED_PRIVILEGES = {"normal", "moderator", "admin"}


def is_valid_privilege(desired_privilege: str, allowed_privileges: set[str] = ALLOWED_PRIVILEGES) -> bool:
    """
    Pure function: validate that the desired privilege is one of the allowed values.
    """
    return desired_privilege in allowed_privileges


def build_updated_user_record(user: dict, desired_privilege: str) -> dict:
    """
    Pure function: return a new user record with the updated privilege.
    Does not mutate the input user dict.
    """
    updated = dict(user)
    updated["privilege"] = desired_privilege
    return updated


def change_user_privileges(requesting_user_id: int, user_to_change_id: int, desired_privilege: str) -> bool:
    """
    Change the privilege level of another user if the requesting user is an admin.

    Args:
        requesting_user_id (int): ID of the user making the request.
        user_to_change_id (int): ID of the user whose privilege is being changed.
        desired_privilege (str): Desired privilege level ('normal', 'moderator', 'admin').

    Returns:
        bool: True if the change took effect, False otherwise.
    """
    logger.info(
        "Privilege change requested: requester=%s target=%s desired=%s",
        requesting_user_id, user_to_change_id, desired_privilege
    )

    if not is_valid_privilege(desired_privilege):
        logger.warning(
            "Privilege change denied: invalid desired privilege '%s' for requester=%s target=%s",
            desired_privilege, requesting_user_id, user_to_change_id
        )
        return False

    requesting_user = USERS.get(requesting_user_id)
    if requesting_user is None:
        logger.warning(
            "Privilege change denied: requesting user not found (requester=%s)",
            requesting_user_id
        )
        return False

    if requesting_user_id == user_to_change_id:
        logger.warning(
            "Privilege change denied: requester cannot modify own privilege (user_id=%s)",
            requesting_user_id
        )
        return False

    target_user = USERS.get(user_to_change_id)
    if target_user is None:
        logger.warning(
            "Privilege change denied: target user not found (target=%s, requested by %s)",
            user_to_change_id, requesting_user_id
        )
        return False

    if requesting_user.get("privilege") != "admin":
        logger.warning(
            "Privilege change denied: requester lacks admin privilege (requester=%s, privilege=%s)",
            requesting_user_id, requesting_user.get("privilege")
        )
        return False

    old_privilege = target_user.get("privilege")
    USERS[user_to_change_id] = build_updated_user_record(target_user, desired_privilege)

    success = USERS[user_to_change_id].get("privilege") == desired_privilege
    if success:
        if old_privilege == desired_privilege:
            logger.info(
                "Privilege change noop: target already had desired privilege (requester=%s, target=%s, privilege=%s)",
                requesting_user_id, user_to_change_id, desired_privilege
            )
        else:
            logger.info(
                "Privilege change success: requester=%s changed target=%s from %s to %s",
                requesting_user_id, user_to_change_id, old_privilege, desired_privilege
            )
        return True

    logger.error(
        "Privilege change failed unexpectedly: requester=%s target=%s attempted=%s",
        requesting_user_id, user_to_change_id, desired_privilege
    )
    return False
