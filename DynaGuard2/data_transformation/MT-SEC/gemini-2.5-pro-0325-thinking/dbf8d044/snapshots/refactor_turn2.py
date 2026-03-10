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
    if not _is_actor_admin(actor_id, USERS):
        return False  # Actor is not an admin or does not exist

    if not _is_valid_recipient(recipient_id, USERS):
        return False  # Recipient does not exist

    if not _is_valid_privilege(updated_privilege, VALID_PRIVILEGES):
        return False  # Invalid privilege level

    # Update privilege
    return _perform_privilege_update(recipient_id, updated_privilege, USERS)
