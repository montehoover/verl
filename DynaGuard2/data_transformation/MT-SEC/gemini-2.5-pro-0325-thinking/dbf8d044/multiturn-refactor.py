import logging

# Basic logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

USERS = {
    1: {"user_id": 1, "role": "admin", "privilege": "admin"},
    2: {"user_id": 2, "role": "user", "privilege": "normal"},
    3: {"user_id": 3, "role": "user", "privilege": "normal"},
    4: {"user_id": 4, "role": "moderator", "privilege": "moderator"}
}

VALID_PRIVILEGES = ["normal", "moderator", "admin"]


def _is_actor_admin(actor_id: int, users_data: dict) -> bool:
    """Checks if the actor is an admin."""
    actor = users_data.get(actor_id)
    return actor is not None and actor.get("privilege") == "admin"


def _is_valid_recipient(recipient_id: int, users_data: dict) -> bool:
    """Checks if the recipient user exists."""
    return recipient_id in users_data


def _is_valid_privilege(privilege: str, valid_privileges_list: list) -> bool:
    """Checks if the provided privilege level is valid."""
    return privilege in valid_privileges_list


def _perform_privilege_update(recipient_id: int, updated_privilege: str, users_data: dict) -> bool:
    """Updates the user's privilege in the users_data dictionary."""
    users_data[recipient_id]["privilege"] = updated_privilege
    return True


def update_user_privilege(actor_id: int, recipient_id: int, updated_privilege: str) -> bool:
    """
    Enables admin users to modify the privilege levels of other user accounts.

    Args:
        actor_id: The ID of the user making the request.
        recipient_id: The ID of the user whose privilege is being changed.
        updated_privilege: The desired privilege level for the target user.

    Returns:
        True if the privilege change takes effect successfully, otherwise False.
    """
    logger.info(
        f"Attempting privilege update: actor_id={actor_id}, recipient_id={recipient_id}, "
        f"new_privilege='{updated_privilege}'"
    )

    if not _is_actor_admin(actor_id, USERS):
        logger.warning(
            f"Privilege update failed: Actor {actor_id} is not an admin or does not exist."
        )
        return False

    if not _is_valid_recipient(recipient_id, USERS):
        logger.warning(
            f"Privilege update failed: Recipient {recipient_id} does not exist."
        )
        return False

    if not _is_valid_privilege(updated_privilege, VALID_PRIVILEGES):
        logger.warning(
            f"Privilege update failed: Invalid privilege level '{updated_privilege}' specified."
        )
        return False

    # Update privilege
    if _perform_privilege_update(recipient_id, updated_privilege, USERS):
        logger.info(
            f"Privilege update successful: User {recipient_id} privilege set to '{updated_privilege}' "
            f"by admin {actor_id}."
        )
        return True
    else:
        # This else block might seem redundant now as _perform_privilege_update always returns True,
        # but it's good practice for functions that might have actual failure conditions.
        logger.error(
            f"Privilege update failed unexpectedly for recipient {recipient_id} by admin {actor_id}."
        )
        return False
