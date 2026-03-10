USERS = {
    1: {"user_id": 1, "role": "admin", "privilege": "admin"},
    2: {"user_id": 2, "role": "user", "privilege": "normal"},
    3: {"user_id": 3, "role": "user", "privilege": "normal"},
    4: {"user_id": 4, "role": "moderator", "privilege": "moderator"}
}

def is_admin(user_id: int) -> bool:
    """
    Checks if a user has admin privileges.

    Args:
        user_id: The ID of the user to check.

    Returns:
        True if the user is an admin, False otherwise.
    """
    user = USERS.get(user_id)
    if user and user.get("role") == "admin":
        return True
    return False

def can_change_privilege(requesting_user_id: int, user_to_change_id: int, desired_privilege: str) -> bool:
    """
    Checks if a user can change another user's privilege.

    Args:
        requesting_user_id: The ID of the user requesting the change.
        user_to_change_id: The ID of the user whose privilege is to be changed.
        desired_privilege: The new privilege level.

    Returns:
        True if the change is permissible, False otherwise.
    """
    if not is_admin(requesting_user_id):
        return False  # Only admins can change privileges

    if user_to_change_id not in USERS:
        return False # User to change must exist

    # For now, any admin can change any existing user's privilege to any value.
    # More specific rules about *which* privileges can be set could be added here.
    return True

def change_user_privileges(requesting_user_id: int, user_to_change_id: int, desired_privilege: str) -> bool:
    """
    Changes a user's privilege level if the requesting user has permission.

    Args:
        requesting_user_id: The ID of the user requesting the change.
        user_to_change_id: The ID of the user whose privilege is to be changed.
        desired_privilege: The new privilege level ('normal', 'moderator', 'admin').

    Returns:
        True if the privilege was successfully changed, False otherwise.
    """
    if not can_change_privilege(requesting_user_id, user_to_change_id, desired_privilege):
        return False

    # Update privilege
    USERS[user_to_change_id]["privilege"] = desired_privilege

    # Update role based on privilege
    if desired_privilege == "admin":
        USERS[user_to_change_id]["role"] = "admin"
    elif desired_privilege == "moderator":
        USERS[user_to_change_id]["role"] = "moderator"
    elif desired_privilege == "normal": # Assuming 'normal' privilege maps to 'user' role
        USERS[user_to_change_id]["role"] = "user"
    else:
        # If desired_privilege is not one of the known values,
        # it might be an error or a new unhandled privilege.
        # For now, we'll not change the role if privilege is unknown.
        # Consider adding error handling or specific logic for new privileges.
        pass # Or raise an error, or set a default role

    return True

if __name__ == '__main__':
    # Example usage for is_admin:
    print(f"User 1 is admin: {is_admin(1)}")  # Expected: True
    print(f"User 2 is admin: {is_admin(2)}")  # Expected: False
    print(f"User 4 is admin: {is_admin(4)}")  # Expected: False
    print(f"User 5 (non-existent) is admin: {is_admin(5)}") # Expected: False

    # Example usage for can_change_privilege:
    # Admin (user 1) changing user 2's privilege to 'moderator'
    print(f"Admin (1) changing user 2 to 'moderator': {can_change_privilege(1, 2, 'moderator')}")  # Expected: True
    # Non-admin (user 2) trying to change user 3's privilege
    print(f"User (2) changing user 3 to 'admin': {can_change_privilege(2, 3, 'admin')}")  # Expected: False
    # Admin (user 1) trying to change a non-existent user (5)
    print(f"Admin (1) changing non-existent user 5 to 'user': {can_change_privilege(1, 5, 'user')}")  # Expected: False
    # Non-admin (user 3) trying to change their own privilege
    print(f"User (3) changing self to 'admin': {can_change_privilege(3, 3, 'admin')}") # Expected: False

    # Example usage for change_user_privileges:
    print(f"\nInitial USERS state: {USERS}")
    # Admin (1) changes user 2's privilege to 'moderator'
    change_success = change_user_privileges(1, 2, "moderator")
    print(f"Admin (1) changing user 2 to 'moderator': Success = {change_success}, New state: {USERS.get(2)}") # Expected: True, role: moderator, privilege: moderator

    # Non-admin (3) tries to change user 2's privilege to 'admin'
    change_success_fail = change_user_privileges(3, 2, "admin")
    print(f"User (3) changing user 2 to 'admin': Success = {change_success_fail}, User 2 state: {USERS.get(2)}") # Expected: False, state unchanged

    # Admin (1) changes user 4's (moderator) privilege to 'normal' (user role)
    change_success_demote = change_user_privileges(1, 4, "normal")
    print(f"Admin (1) changing user 4 to 'normal': Success = {change_success_demote}, New state: {USERS.get(4)}") # Expected: True, role: user, privilege: normal

    # Admin (1) changes user 3's privilege to 'admin'
    change_success_promote_admin = change_user_privileges(1, 3, "admin")
    print(f"Admin (1) changing user 3 to 'admin': Success = {change_success_promote_admin}, New state: {USERS.get(3)}") # Expected: True, role: admin, privilege: admin

    # Admin (1) tries to change non-existent user 5
    change_success_non_existent = change_user_privileges(1, 5, "user")
    print(f"Admin (1) changing non-existent user 5: Success = {change_success_non_existent}") # Expected: False
    print(f"Final USERS state: {USERS}")
