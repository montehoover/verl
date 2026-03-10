# Global data structure for user roles and privileges
USERS = {
    1: {"user_id": 1, "role": "admin", "privilege": "admin"},
    2: {"user_id": 2, "role": "user", "privilege": "normal"},
    3: {"user_id": 3, "role": "user", "privilege": "normal"},
    4: {"user_id": 4, "role": "moderator", "privilege": "moderator"}
    # Add more users and their privileges here
}

def get_user_privilege(user_id: int) -> str:
    """
    Checks the current privilege level of a user.

    Args:
        user_id: The ID of the user.

    Returns:
        The privilege level of the user as a string,
        or "unknown" if the user is not found.
    """
    user_data = USERS.get(user_id)
    if user_data:
        return user_data.get("privilege", "unknown")
    return "unknown"

def generate_user_report() -> list:
    """
    Generates a list of all user IDs and their privilege levels.

    Returns:
        A list of dictionaries, where each dictionary contains
        'user_id' and 'privilege'.
    """
    report = []
    for user_id, user_data in USERS.items():
        report.append({'user_id': user_id, 'privilege': user_data.get("privilege", "unknown")})
    return report

def modify_user_privileges(current_user_id: int, target_user_id: int, new_privilege: str) -> bool:
    """
    Allows an admin user to modify the privileges of a target user.

    Args:
        current_user_id: The ID of the user attempting the change.
        target_user_id: The ID of the user whose privileges are being modified.
        new_privilege: The new privilege level to assign.

    Returns:
        True if the privilege was successfully modified, False otherwise.
    """
    # Check if the current user exists and is an admin
    current_user_data = USERS.get(current_user_id)
    if not current_user_data or current_user_data.get("privilege") != "admin":
        return False

    # Check if the target user exists
    target_user_data = USERS.get(target_user_id)
    if not target_user_data:
        return False

    # Modify the privilege
    USERS[target_user_id]["privilege"] = new_privilege
    return True
