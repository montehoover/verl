import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

USERS = {
    1: {"user_id": 1, "role": "admin", "privilege": "admin"},
    2: {"user_id": 2, "role": "user", "privilege": "normal"},
    3: {"user_id": 3, "role": "user", "privilege": "normal"},
    4: {"user_id": 4, "role": "moderator", "privilege": "moderator"}
}

VALID_PRIVILEGES = ["normal", "moderator", "admin"]


def _is_valid_privilege(privilege: str) -> bool:
    """Checks if the desired privilege is valid."""
    return privilege in VALID_PRIVILEGES


def _commit_privilege_change(user_id: int, new_privilege: str):
    """Updates the user's privilege in the USERS dictionary."""
    USERS[user_id]["privilege"] = new_privilege


def change_user_privileges(requesting_user_id: int, user_to_change_id: int, desired_privilege: str) -> bool:
    """
    Enables admin users to modify the privilege levels of other user accounts.

    Args:
        requesting_user_id: The ID of the user making the request.
        user_to_change_id: The ID of the user whose privilege is being changed.
        desired_privilege: The desired privilege level for the target user.

    Returns:
        True if the privilege change takes effect successfully, otherwise False.
    """
    logging.info(
        f"User {requesting_user_id} attempting to change privilege of user {user_to_change_id} to '{desired_privilege}'."
    )

    # Check if the requesting user exists and is an admin
    requesting_user = USERS.get(requesting_user_id)
    if not requesting_user:
        logging.warning(
            f"Privilege change failed: Requesting user {requesting_user_id} not found."
        )
        return False
    
    if requesting_user.get("privilege") != "admin":
        logging.warning(
            f"Privilege change failed: User {requesting_user_id} is not an admin."
        )
        return False

    # Check if the user to change exists
    user_to_change = USERS.get(user_to_change_id)
    if not user_to_change:
        logging.warning(
            f"Privilege change failed: Target user {user_to_change_id} not found."
        )
        return False

    # Check if the desired privilege is valid
    if not _is_valid_privilege(desired_privilege):
        logging.warning(
            f"Privilege change failed: Desired privilege '{desired_privilege}' is invalid for user {user_to_change_id}."
        )
        return False

    # Update the privilege
    _commit_privilege_change(user_to_change_id, desired_privilege)
    logging.info(
        f"Successfully changed privilege of user {user_to_change_id} to '{desired_privilege}' by user {requesting_user_id}."
    )
    return True
