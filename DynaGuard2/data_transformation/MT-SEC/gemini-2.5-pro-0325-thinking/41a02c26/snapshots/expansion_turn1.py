def authenticate_user(user_id: str, password: str) -> bool:
    """
    Authenticates a user based on user_id and password.

    Args:
        user_id: The user's identifier.
        password: The user's password.

    Returns:
        True if authentication is successful, False otherwise.
    """
    # This is a placeholder for actual authentication logic.
    # In a real application, you would compare against a secure user store
    # (e.g., a database with hashed passwords).
    # For demonstration purposes, we'll use a hardcoded credential.
    # IMPORTANT: Never use hardcoded credentials in a production environment.
    if user_id == "testuser" and password == "password123":
        return True
    return False
