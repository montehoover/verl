def check_user_existence(user_id: str) -> bool:
    """
    Verifies if a user exists in a system's database.

    Args:
        user_id: The ID of the user to check.

    Returns:
        True if the user exists, False otherwise.
    """
    # This is a placeholder for actual database lookup logic.
    # In a real application, you would query your database here.
    # For example:
    # db_users = {"user123": {"name": "Alice"}, "user456": {"name": "Bob"}}
    # return user_id in db_users
    print(f"Checking existence of user: {user_id}")  # Placeholder action
    if user_id == "existing_user":  # Simulate an existing user
        return True
    return False

if __name__ == '__main__':
    # Example Usage
    user1 = "existing_user"
    user2 = "non_existing_user"

    print(f"Does user '{user1}' exist? {check_user_existence(user1)}")
    print(f"Does user '{user2}' exist? {check_user_existence(user2)}")
