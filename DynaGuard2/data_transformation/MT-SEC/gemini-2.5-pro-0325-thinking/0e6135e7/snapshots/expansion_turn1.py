def authenticate_user(user_id: str, password: str) -> bool:
    """
    Authenticates a user based on their user_id and password.

    Args:
        user_id: The user's identifier.
        password: The user's password.

    Returns:
        True if authentication is successful, False otherwise.
    """
    # This is a placeholder for actual authentication logic.
    # In a real application, you would compare against securely stored credentials.
    # For example, fetching a hashed password from a database and comparing it.
    if user_id == "test_user" and password == "secure_password123":
        return True
    return False
