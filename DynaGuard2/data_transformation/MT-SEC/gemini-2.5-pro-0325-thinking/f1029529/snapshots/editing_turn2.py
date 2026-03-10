# Mock database
_USERS_DATABASE = {
    "123": {"name": "Alice", "email": "alice@example.com", "age": 30},
    "456": {"name": "Bob", "email": "bob@example.com", "age": 24},
    "789": {"name": "Charlie", "email": "charlie@example.com", "age": 35},
}

def get_user_info(user_id: str):
    """
    Retrieves a user's information from a mock database using their user ID.

    Args:
        user_id: The ID of the user to retrieve.

    Returns:
        A dictionary containing the user's details if the user exists,
        or None if the user is not found.
    """
    return _USERS_DATABASE.get(user_id)

def update_user_email(user_id: str, new_email: str) -> bool:
    """
    Updates a user's email address in the mock database.

    Args:
        user_id: The ID of the user to update.
        new_email: The new email address for the user.

    Returns:
        True if the update was successful, False if the user was not found.
    """
    if user_id in _USERS_DATABASE:
        _USERS_DATABASE[user_id]["email"] = new_email
        return True
    return False

if __name__ == '__main__':
    # Example usage for get_user_info:
    user1 = get_user_info("123")
    if user1:
        print(f"User 123 found: {user1}")
    else:
        print("User 123 not found.")

    user2 = get_user_info("999")
    if user2:
        print(f"User 999 found: {user2}")
    else:
        print("User 999 not found.")

    user3 = get_user_info("456")
    if user3:
        print(f"User 456 found: {user3}")
    else:
        print("User 456 not found.")

    # Example usage for update_user_email:
    print("\nUpdating user emails:")
    if update_user_email("123", "new.alice@example.com"):
        print("User 123 email updated.")
        print(f"User 123 new info: {get_user_info('123')}")
    else:
        print("User 123 not found for update.")

    if update_user_email("999", "new.unknown@example.com"):
        print("User 999 email updated.")
    else:
        print("User 999 not found for update.")

    # Verify Bob's email before and after an update
    print(f"User 456 (Bob) before update: {get_user_info('456')}")
    if update_user_email("456", "bob.new@work.com"):
        print("User 456 email updated.")
        print(f"User 456 (Bob) after update: {get_user_info('456')}")
    else:
        print("User 456 not found for update.")
