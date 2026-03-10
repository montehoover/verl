# Global data structure for user roles
# Example: {1: 'admin', 2: 'user', 3: 'admin'}
USER_ROLES = {}

def check_user_role(user_id: int) -> str:
    """
    Determines if a user is an admin or a regular user.

    Args:
        user_id: The ID of the user.

    Returns:
        'admin' if the user is an admin, or 'user' if they are a regular user.
        Returns 'user' by default if the user_id is not found or has no specific role.
    """
    role = USER_ROLES.get(user_id)
    if role == 'admin':
        return 'admin'
    return 'user'
