def authenticate_personnel(personnel_id: str, password: str) -> bool:
    """
    Authenticates personnel based on their ID and password.

    Args:
        personnel_id: The ID of the personnel.
        password: The password of the personnel.

    Returns:
        True if the credentials are valid, False otherwise.
    """
    # This is a placeholder for actual authentication logic.
    # In a real application, you would check against a database
    # or an authentication service.
    # For demonstration purposes, let's assume a fixed set of credentials.
    valid_credentials = {
        "user123": "password123",
        "admin001": "securePass!"
    }

    if personnel_id in valid_credentials and valid_credentials[personnel_id] == password:
        return True
    return False

if __name__ == '__main__':
    # Example Usage
    test_user1_valid = authenticate_personnel("user123", "password123")
    print(f"Authentication for user123 (valid): {test_user1_valid}") # Expected: True

    test_user1_invalid_pass = authenticate_personnel("user123", "wrongpassword")
    print(f"Authentication for user123 (invalid password): {test_user1_invalid_pass}") # Expected: False

    test_unknown_user = authenticate_personnel("unknown_user", "password123")
    print(f"Authentication for unknown_user: {test_unknown_user}") # Expected: False

    test_admin_valid = authenticate_personnel("admin001", "securePass!")
    print(f"Authentication for admin001 (valid): {test_admin_valid}") # Expected: True
