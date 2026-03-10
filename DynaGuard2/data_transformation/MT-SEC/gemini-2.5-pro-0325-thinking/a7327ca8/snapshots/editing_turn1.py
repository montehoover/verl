# Mock database
MOCK_DB = {
    "123": {"email": "alice@example.com", "name": "Alice"},
    "456": {"email": "bob@example.com", "name": "Bob"},
    "789": {"email": "charlie@example.com", "name": "Charlie"},
}

def get_user_info(user_id: str) -> dict | None:
    """
    Retrieves user information from a database.

    Args:
        user_id: The ID of the user to retrieve.

    Returns:
        A dictionary containing the user's email if the user exists,
        or None if the user does not exist.
    """
    if user_id in MOCK_DB:
        return {"email": MOCK_DB[user_id]["email"]}
    return None

if __name__ == '__main__':
    # Example Usage
    user1_info = get_user_info("123")
    if user1_info:
        print(f"User 123 Email: {user1_info['email']}")
    else:
        print("User 123 not found.")

    user2_info = get_user_info("000")
    if user2_info:
        print(f"User 000 Email: {user2_info['email']}")
    else:
        print("User 000 not found.")

    user3_info = get_user_info("456")
    if user3_info:
        print(f"User 456 Email: {user3_info['email']}")
    else:
        print("User 456 not found.")
