def authenticate_user(staff_id: str, password: str) -> bool:
    """
    Authenticates a user based on staff_id and password.

    Args:
        staff_id: The staff ID of the user.
        password: The password of the user.

    Returns:
        True if the credentials are correct, False otherwise.
    """
    # In a real application, you would look up the staff_id
    # in a database and compare the hashed password.
    # For this example, we'll use a hardcoded credential.
    if staff_id == "admin" and password == "password123":
        return True
    return False
