# For demonstration purposes, using a hardcoded user and password.
# In a real application, this would involve checking against a database
# or another secure user store, and passwords would be hashed.
_users_credentials = {
    "admin": "password123"
}

# For demonstration purposes, using a hardcoded user roles.
# In a real application, this would involve checking against a database
# or another secure user store.
_user_roles = {
    "admin": "administrator",
    "editor1": "editor",
    "viewer1": "viewer"
}

def authenticate_user(user_id: str, password: str) -> bool:
    """
    Authenticates a user based on their user_id and password.

    Args:
        user_id: The user's identifier.
        password: The user's password.

    Returns:
        True if authentication is successful, False otherwise.
    """
    if user_id in _users_credentials and _users_credentials[user_id] == password:
        return True
    return False

def check_user_role(user_id: str) -> str:
    """
    Retrieves the role of a user.

    Args:
        user_id: The user's identifier.

    Returns:
        The user's role as a string, or an empty string if the user is not found or has no role.
    """
    return _user_roles.get(user_id, "")
