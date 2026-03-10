USERS = {
    1: {"user_id": 1, "role": "admin", "privilege": "admin"},
    2: {"user_id": 2, "role": "user", "privilege": "normal"},
    3: {"user_id": 3, "role": "user", "privilege": "normal"},
    4: {"user_id": 4, "role": "moderator", "privilege": "moderator"}
}

VALID_PRIVILEGES = {'normal', 'moderator', 'admin'}


def _user_exists(user_id: int) -> bool:
    """Checks if a user exists in the USERS dictionary."""
    return user_id in USERS


def _is_user_admin(user_id: int) -> bool:
    """Checks if the user has admin privileges."""
    if not _user_exists(user_id):
        return False
    return USERS[user_id].get("privilege") == "admin"


def _is_valid_privilege(privilege: str) -> bool:
    """Checks if the given privilege is valid."""
    return privilege in VALID_PRIVILEGES


def _set_user_privilege(user_id: int, new_privilege: str) -> None:
    """Sets the privilege for the specified user."""
    USERS[user_id]["privilege"] = new_privilege
    # Optionally, if role and privilege should always match:
    # USERS[user_id]["role"] = new_privilege


def update_privileges(acting_user_id: int, affected_user_id: int, new_role: str) -> bool:
    """
    Facilitates the modification of user privileges by administrators.

    Args:
        acting_user_id: The ID of the user attempting to change privileges.
        affected_user_id: The ID of the user whose privilege is being altered.
        new_role: The new privilege level designated for that user.

    Returns:
        True if the privilege modification is successful, otherwise False.
    """
    if not _is_user_admin(acting_user_id):
        return False

    if not _user_exists(affected_user_id):
        return False

    if not _is_valid_privilege(new_role):
        return False

    _set_user_privilege(affected_user_id, new_role)
    return True


if __name__ == '__main__':
    # Example Usage:
    print(f"Initial USERS: {USERS}")

    # Admin (1) tries to change user (2)'s privilege to 'moderator'
    success = update_privileges(acting_user_id=1, affected_user_id=2, new_role="moderator")
    print(f"Admin (1) changing user (2) to moderator: {'Success' if success else 'Failed'}")
    print(f"USERS after change: {USERS}")

    # Non-admin (2) tries to change user (3)'s privilege
    success = update_privileges(acting_user_id=2, affected_user_id=3, new_role="admin")
    print(f"User (2) changing user (3) to admin: {'Success' if success else 'Failed'}")
    print(f"USERS after change: {USERS}")

    # Admin (1) tries to change non-existent user (5)'s privilege
    success = update_privileges(acting_user_id=1, affected_user_id=5, new_role="moderator")
    print(f"Admin (1) changing non-existent user (5): {'Success' if success else 'Failed'}")
    print(f"USERS after change: {USERS}")

    # Admin (1) tries to set an invalid privilege for user (3)
    success = update_privileges(acting_user_id=1, affected_user_id=3, new_role="super_user")
    print(f"Admin (1) setting invalid privilege for user (3): {'Success' if success else 'Failed'}")
    print(f"USERS after change: {USERS}")

    # Admin (1) successfully changes user (3)'s privilege to 'admin'
    success = update_privileges(acting_user_id=1, affected_user_id=3, new_role="admin")
    print(f"Admin (1) changing user (3) to admin: {'Success' if success else 'Failed'}")
    print(f"USERS after change: {USERS}")
