def get_user_info(user_id: str) -> dict:
    """
    Retrieves a user's information from a mock database.

    Args:
        user_id: The ID of the user to retrieve.

    Returns:
        A dictionary containing the user's details (name, phone number),
        or an empty dictionary if the user is not found.
    """
    # Mock database
    mock_db = {
        "123": {"name": "Alice Wonderland", "phone_number": "555-0101"},
        "456": {"name": "Bob The Builder", "phone_number": "555-0202"},
        "789": {"name": "Charlie Brown", "phone_number": "555-0303"},
    }

    return mock_db.get(user_id, {})

if __name__ == '__main__':
    # Example usage:
    user1_info = get_user_info("123")
    print(f"User 123 info: {user1_info}")

    user2_info = get_user_info("456")
    print(f"User 456 info: {user2_info}")

    user_not_found = get_user_info("999")
    print(f"User 999 info: {user_not_found}")
