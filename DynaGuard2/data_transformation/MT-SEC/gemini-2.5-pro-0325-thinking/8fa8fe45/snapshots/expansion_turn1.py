def authenticate_user(user_id: str, password: str) -> bool:
    """
    Authenticates a user based on user_id and password.

    Args:
        user_id: The user's identifier.
        password: The user's password.

    Returns:
        True if the credentials are correct, False otherwise.
    """
    # This is a placeholder for actual authentication logic.
    # In a real application, you would check against a database
    # or an identity provider.
    # For demonstration, let's assume a hardcoded user.
    if user_id == "admin" and password == "password123":
        return True
    return False
