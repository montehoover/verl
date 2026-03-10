# Global dictionary to store user information
users_db = {}

def add_user(user_id: str, user_info: dict) -> bool:
    """
    Adds a user to the global user dictionary.

    Args:
        user_id: The ID of the user (string).
        user_info: A dictionary containing user details (e.g., 'name', 'email').

    Returns:
        True if the user was added successfully, False otherwise.
    """
    # Check if user_id already exists to prevent overwriting,
    # or decide if overwriting is the desired behavior.
    # For now, let's allow overwriting.
    try:
        users_db[user_id] = user_info
        return True
    except Exception:
        # In a more complex scenario, you might have specific error handling
        return False

if __name__ == '__main__':
    # Example usage:
    user1_info = {"name": "Alice Wonderland", "email": "alice@example.com"}
    added1 = add_user("user123", user1_info)
    print(f"User 'user123' added: {added1}")

    user2_info = {"name": "Bob The Builder", "email": "bob@example.com"}
    added2 = add_user("user456", user2_info)
    print(f"User 'user456' added: {added2}")

    print("\nCurrent users in DB:")
    for user_id, info in users_db.items():
        print(f"ID: {user_id}, Info: {info}")
