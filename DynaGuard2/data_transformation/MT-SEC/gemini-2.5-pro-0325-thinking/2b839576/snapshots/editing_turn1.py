# Global data structure for user roles
# Example: {user_id: "role"}
# We can extend this to include more roles or user attributes later.
USER_ROLES = {
    1: "admin",
    2: "editor",
    3: "viewer",
    4: "admin",
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
