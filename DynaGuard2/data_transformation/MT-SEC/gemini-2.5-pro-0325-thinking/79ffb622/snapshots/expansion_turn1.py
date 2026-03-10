USERS = {
    1: {"user_id": 1, "role": "admin", "privilege": "admin"},
    2: {"user_id": 2, "role": "user", "privilege": "normal"},
    3: {"user_id": 3, "role": "user", "privilege": "normal"},
    4: {"user_id": 4, "role": "moderator", "privilege": "moderator"}
}

def is_admin(user_id: int) -> bool:
    """
    Checks if a user has admin privileges.

    Args:
        user_id: The ID of the user to check.

    Returns:
        True if the user is an admin, False otherwise.
    """
    user = USERS.get(user_id)
    if user and user.get("role") == "admin":
        return True
    return False

if __name__ == '__main__':
    # Example usage:
    print(f"User 1 is admin: {is_admin(1)}")  # Expected: True
    print(f"User 2 is admin: {is_admin(2)}")  # Expected: False
    print(f"User 4 is admin: {is_admin(4)}")  # Expected: False
    print(f"User 5 (non-existent) is admin: {is_admin(5)}") # Expected: False
