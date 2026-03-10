def authenticate_user(user_id: str, password: str) -> bool:
    """
    Authenticates a user based on their user_id and password.

    Args:
        user_id: The user's identifier.
        password: The user's password.

    Returns:
        True if the credentials are correct, False otherwise.
    """
    # In a real application, you would look up the user_id in a database
    # and compare the hashed password.
    # For this example, we'll use a placeholder.
    # Replace this with actual authentication logic.
    if user_id == "admin" and password == "password123":
        return True
    return False
