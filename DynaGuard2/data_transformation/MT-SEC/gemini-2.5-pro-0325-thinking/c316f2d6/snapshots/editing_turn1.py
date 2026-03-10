# Global data structure for user roles
# This is a simplified example. In a real application, this might come from a database.
USER_ROLES = {
    1: "admin",
    2: "editor",
    3: "viewer",
    4: "admin",
    5: "editor",
}

def is_user_admin(user_id: int) -> bool:
    """
    Checks if a given user is an admin.

    Args:
        user_id: The ID of the user to check.

    Returns:
        True if the user is an admin, False otherwise.
    """
    return USER_ROLES.get(user_id) == "admin"
