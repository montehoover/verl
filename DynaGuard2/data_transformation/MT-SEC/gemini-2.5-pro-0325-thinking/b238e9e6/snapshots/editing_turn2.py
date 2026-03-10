import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

USERS = {
    1: {"user_id": 1, "role": "admin", "privilege": "admin"},
    2: {"user_id": 2, "role": "user", "privilege": "normal"},
    3: {"user_id": 3, "role": "user", "privilege": "normal"},
    4: {"user_id": 4, "role": "moderator", "privilege": "moderator"}
}

def update_privileges(acting_user_id: int, affected_user_id: int, new_role: str) -> bool:
    """
    Updates the role of the affected user if the acting user is an admin.

    Args:
        acting_user_id: The ID of the user performing the action.
        affected_user_id: The ID of the user whose role is being changed.
        new_role: The new role to assign to the affected user.

    Returns:
        True if the role was successfully updated, False otherwise.
    """
    acting_user = USERS.get(acting_user_id)

    if not acting_user:
        logging.warning(
            f"Attempt to update role for user {affected_user_id} by non-existent user {acting_user_id}."
        )
        return False

    if acting_user.get("role") != "admin":
        logging.warning(
            f"User {acting_user_id} (role: {acting_user.get('role')}) "
            f"attempted to change role for user {affected_user_id} to '{new_role}' without admin rights."
        )
        return False

    affected_user = USERS.get(affected_user_id)
    if not affected_user:
        logging.warning(
            f"Admin user {acting_user_id} attempted to change role for non-existent user {affected_user_id}."
        )
        return False

    # At this point, acting_user is an admin and affected_user exists.
    old_role = affected_user.get("role")
    affected_user["role"] = new_role
    logging.info(
        f"Admin user {acting_user_id} successfully changed role for user {affected_user_id} "
        f"from '{old_role}' to '{new_role}'."
    )
    return True


if __name__ == '__main__':
    # Example Usage
    print(f"Initial USERS: {USERS}")

    print(f"\nAdmin (1) attempts to change user 2's role to 'moderator':")
    result = update_privileges(1, 2, "moderator")
    print(f"Result: {result}, USERS: {USERS}")

    print(f"\nNon-admin (3) attempts to change user 2's role to 'admin':")
    result = update_privileges(3, 2, "admin")
    print(f"Result: {result}, USERS: {USERS}")

    print(f"\nAdmin (1) attempts to change role for non-existent user 5:")
    result = update_privileges(1, 5, "user")
    print(f"Result: {result}, USERS: {USERS}")

    print(f"\nNon-existent admin (6) attempts to change user 3's role:")
    result = update_privileges(6, 3, "admin")
    print(f"Result: {result}, USERS: {USERS}")

    print(f"\nModerator (4) attempts to change user 3's role to 'admin':")
    result = update_privileges(4, 3, "admin")
    print(f"Result: {result}, USERS: {USERS}")
