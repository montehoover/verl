def check_user_exists(user_id: str) -> bool:
    """
    Verifies if a user exists in a database.

    Args:
        user_id: The ID of the user to check.

    Returns:
        True if the user exists, False otherwise.
    """
    # This is a placeholder for actual database lookup logic.
    # In a real application, you would query your database here.
    # For example:
    # db_users = {"user1": "Alice", "user2": "Bob"}
    # return user_id in db_users
    print(f"Checking if user '{user_id}' exists...")  # Placeholder action
    if user_id == "existing_user":
        return True
    return False

if __name__ == '__main__':
    # Example usage:
    user1_exists = check_user_exists("existing_user")
    print(f"User 'existing_user' exists: {user1_exists}")

    user2_exists = check_user_exists("non_existing_user")
    print(f"User 'non_existing_user' exists: {user2_exists}")
