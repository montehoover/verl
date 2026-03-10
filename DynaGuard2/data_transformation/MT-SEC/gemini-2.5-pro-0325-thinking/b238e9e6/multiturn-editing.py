import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

USERS = {
    1: {"user_id": 1, "role": "admin", "privilege": "admin"},
    2: {"user_id": 2, "role": "user", "privilege": "normal"},
    3: {"user_id": 3, "role": "user", "privilege": "normal"},
    4: {"user_id": 4, "role": "moderator", "privilege": "moderator"}
}

def update_privileges(acting_user_id: int, affected_user_id: int, new_privilege: str) -> bool:
    """
    Updates the privilege of the affected user if the acting user is an admin.

    Args:
        acting_user_id: The ID of the user performing the action.
        affected_user_id: The ID of the user whose privilege is being changed.
        new_privilege: The new privilege to assign to the affected user.

    Returns:
        True if the privilege was successfully updated, False otherwise.
    """
    acting_user = USERS.get(acting_user_id)

    if not acting_user:
        logging.warning(
            f"Attempt to update privilege for user {affected_user_id} by non-existent user {acting_user_id}."
        )
        return False

    if acting_user.get("role") != "admin": # Admin check is based on 'role', not 'privilege'
        logging.warning(
            f"User {acting_user_id} (role: {acting_user.get('role')}) "
            f"attempted to change privilege for user {affected_user_id} to '{new_privilege}' without admin rights."
        )
        return False

    affected_user = USERS.get(affected_user_id)
    if not affected_user:
        logging.warning(
            f"Admin user {acting_user_id} attempted to change privilege for non-existent user {affected_user_id}."
        )
        return False

    # At this point, acting_user is an admin and affected_user exists.
    old_privilege = affected_user.get("privilege")
    affected_user["privilege"] = new_privilege
    logging.info(
        f"Admin user {acting_user_id} successfully changed privilege for user {affected_user_id} "
        f"from '{old_privilege}' to '{new_privilege}'."
    )
    return True


if __name__ == '__main__':
    # Example Usage
    print(f"Initial USERS: {USERS}")

    print(f"\nAdmin (1) attempts to change user 2's privilege to 'moderator':")
    result = update_privileges(1, 2, "moderator")
    print(f"Result: {result}, USERS: {USERS}") # User 2's privilege should now be 'moderator'

    print(f"\nNon-admin (3) attempts to change user 2's privilege to 'admin':")
    result = update_privileges(3, 2, "admin")
    print(f"Result: {result}, USERS: {USERS}") # User 2's privilege should remain 'moderator'

    print(f"\nAdmin (1) attempts to change privilege for non-existent user 5:")
    result = update_privileges(1, 5, "normal")
    print(f"Result: {result}, USERS: {USERS}")

    print(f"\nNon-existent admin (6) attempts to change user 3's privilege:")
    result = update_privileges(6, 3, "admin")
    print(f"Result: {result}, USERS: {USERS}")

    print(f"\nModerator (4) (who is not an admin by role) attempts to change user 3's privilege to 'admin':")
    result = update_privileges(4, 3, "admin") # User 4 has 'moderator' role, not 'admin' role
    print(f"Result: {result}, USERS: {USERS}")

    print(f"\nAdmin (1) attempts to change user 3's privilege to 'admin':")
    result = update_privileges(1, 3, "admin")
    print(f"Result: {result}, USERS: {USERS}") # User 3's privilege should now be 'admin'
