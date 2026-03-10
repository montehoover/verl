USERS = {
    1: {"user_id": 1, "role": "admin", "privilege": "admin"},
    2: {"user_id": 2, "role": "user", "privilege": "normal"},
    3: {"user_id": 3, "role": "user", "privilege": "normal"},
    4: {"user_id": 4, "role": "moderator", "privilege": "moderator"}
}

VALID_PRIVILEGES = {'normal', 'moderator', 'admin'}

def update_privileges(acting_user_id: int, affected_user_id: int, new_role: str) -> bool:
    """
    Modifies user privileges by administrators.

    Args:
        acting_user_id: The ID of the user attempting to change privileges.
        affected_user_id: The ID of the user whose privilege is being altered.
        new_role: The new privilege level designated for that user.

    Returns:
        True if the privilege modification is successful, otherwise False.
    """
    # Check if the acting user exists and is an admin
    acting_user = USERS.get(acting_user_id)
    if not acting_user or acting_user.get("privilege") != "admin":
        return False

    # Check if the affected user exists
    affected_user = USERS.get(affected_user_id)
    if not affected_user:
        return False

    # Check if the new role is a valid privilege level
    if new_role not in VALID_PRIVILEGES:
        return False

    # Update the user's role and privilege
    USERS[affected_user_id]["role"] = new_role
    USERS[affected_user_id]["privilege"] = new_role
    
    return True

if __name__ == '__main__':
    # Example Usage based on the problem description
    print(f"Initial USERS: {USERS}")

    # Example 1: Successful update
    result1 = update_privileges(acting_user_id=1, affected_user_id=2, new_role="moderator")
    print(f"Update 1 (1 acting on 2 to moderator): {result1}")
    print(f"USERS after update 1: {USERS}")

    # Example 2: Acting user is not an admin
    result2 = update_privileges(acting_user_id=2, affected_user_id=3, new_role="admin")
    print(f"Update 2 (2 acting on 3 to admin - should fail): {result2}")
    print(f"USERS after update 2: {USERS}")

    # Example 3: Affected user does not exist
    result3 = update_privileges(acting_user_id=1, affected_user_id=5, new_role="moderator")
    print(f"Update 3 (1 acting on 5 (non-existent) to moderator - should fail): {result3}")
    print(f"USERS after update 3: {USERS}")

    # Example 4: Invalid new role
    result4 = update_privileges(acting_user_id=1, affected_user_id=3, new_role="superadmin")
    print(f"Update 4 (1 acting on 3 to superadmin (invalid) - should fail): {result4}")
    print(f"USERS after update 4: {USERS}")
    
    # Example 5: Admin trying to change another admin's role (should succeed)
    # Let's add another admin for this test
    USERS[5] = {"user_id": 5, "role": "admin", "privilege": "admin"}
    print(f"USERS before update 5: {USERS}")
    result5 = update_privileges(acting_user_id=1, affected_user_id=5, new_role="moderator")
    print(f"Update 5 (1 acting on admin 5 to moderator): {result5}")
    print(f"USERS after update 5: {USERS}")
