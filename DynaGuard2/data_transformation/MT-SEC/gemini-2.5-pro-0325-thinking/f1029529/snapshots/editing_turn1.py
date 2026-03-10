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

if __name__ == '__main__':
    # Example usage:
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
