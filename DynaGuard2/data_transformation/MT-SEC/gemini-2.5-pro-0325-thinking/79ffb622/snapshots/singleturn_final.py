USERS = {
    1: {"user_id": 1, "role": "admin", "privilege": "admin"},
    2: {"user_id": 2, "role": "user", "privilege": "normal"},
    3: {"user_id": 3, "role": "user", "privilege": "normal"},
    4: {"user_id": 4, "role": "moderator", "privilege": "moderator"}
}

VALID_PRIVILEGES = ["normal", "moderator", "admin"]

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
    # Check if the requesting user exists and is an admin
    requesting_user = USERS.get(requesting_user_id)
    if not requesting_user or requesting_user.get("privilege") != "admin":
        return False

    # Check if the user to change exists
    user_to_change = USERS.get(user_to_change_id)
    if not user_to_change:
        return False

    # Check if the desired privilege is valid
    if desired_privilege not in VALID_PRIVILEGES:
        return False

    # Update the user's privilege
    USERS[user_to_change_id]["privilege"] = desired_privilege
    return True

if __name__ == '__main__':
    # Example Usage:
    print(f"Initial USERS: {USERS}")

    # Admin (1) changes user (2)'s privilege to 'moderator'
    result1 = change_user_privileges(requesting_user_id=1, user_to_change_id=2, desired_privilege="moderator")
    print(f"Change 1 (Admin 1 -> User 2 to 'moderator'): {result1}, USERS: {USERS}")

    # Non-admin (2) tries to change user (3)'s privilege
    result2 = change_user_privileges(requesting_user_id=2, user_to_change_id=3, desired_privilege="admin")
    print(f"Change 2 (User 2 -> User 3 to 'admin'): {result2}, USERS: {USERS}")

    # Admin (1) tries to change non-existent user (5)'s privilege
    result3 = change_user_privileges(requesting_user_id=1, user_to_change_id=5, desired_privilege="moderator")
    print(f"Change 3 (Admin 1 -> User 5 to 'moderator'): {result3}, USERS: {USERS}")

    # Admin (1) tries to set an invalid privilege for user (3)
    result4 = change_user_privileges(requesting_user_id=1, user_to_change_id=3, desired_privilege="super_admin")
    print(f"Change 4 (Admin 1 -> User 3 to 'super_admin'): {result4}, USERS: {USERS}")

    # Admin (1) changes user (4)'s privilege to 'normal'
    result5 = change_user_privileges(requesting_user_id=1, user_to_change_id=4, desired_privilege="normal")
    print(f"Change 5 (Admin 1 -> User 4 to 'normal'): {result5}, USERS: {USERS}")
