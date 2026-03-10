# A simple in-memory store for users and their roles
# In a real application, this would be a database or an external auth system.
_users_roles = {
    "user123": "admin",
    "user456": "editor",
    "user789": "viewer",
}

def authenticate_user(user_id: str, role: str) -> bool:
    """
    Authenticates a user based on their user_id and role.

    Args:
        user_id: The ID of the user.
        role: The role to check for the user.

    Returns:
        True if the user has the specified role, False otherwise.
    """
    if user_id in _users_roles and _users_roles[user_id] == role:
        return True
    return False

if __name__ == '__main__':
    # Example Usage
    print(f"Authenticating user123 as admin: {authenticate_user('user123', 'admin')}")
    print(f"Authenticating user123 as editor: {authenticate_user('user123', 'editor')}")
    print(f"Authenticating user456 as editor: {authenticate_user('user456', 'editor')}")
    print(f"Authenticating user789 as viewer: {authenticate_user('user789', 'viewer')}")
    print(f"Authenticating unknown_user as admin: {authenticate_user('unknown_user', 'admin')}")
    print(f"Authenticating user123 as unknown_role: {authenticate_user('user123', 'unknown_role')}")
