def authenticate_staff(staff_id: str, password: str) -> bool:
    """
    Authenticates healthcare staff based on their ID and password.

    Args:
        staff_id: The staff member's unique identifier.
        password: The staff member's password.

    Returns:
        True if the credentials are valid, False otherwise.
    """
    # In a real application, this would involve checking credentials
    # against a secure database or an authentication service.
    # For this example, we'll use a placeholder implementation.
    if staff_id == "admin" and password == "password123":
        return True
    return False
