# Global data structure for user roles
# Example: {user_id: role}
USER_ROLES = {
    1: 'admin',
    2: 'editor',
    3: 'viewer',
}

def can_access_content(user_id: int) -> bool:
    """
    Checks if a user can access any content based on their role.
    Returns True if the user is an admin, otherwise False.
    """
    role = USER_ROLES.get(user_id)
    return role == 'admin'
