# Global list to store user information
users_db = []

def add_user(user_id: str, user_info: str) -> bool:
    """
    Adds a user to the global user list.

    Args:
        user_id: The ID of the user (string).
        user_info: Information about the user (string).

    Returns:
        True if the user was added successfully, False otherwise.
    """
    try:
        # For simplicity, we'll store user info as a dictionary
        # You could also use a tuple: (user_id, user_info)
        users_db.append({"id": user_id, "info": user_info})
        return True
    except Exception:
        # In a more complex scenario, you might have specific error handling
        return False

if __name__ == '__main__':
    # Example usage:
    added1 = add_user("user123", "Alice Wonderland, alice@example.com")
    print(f"User 'user123' added: {added1}")

    added2 = add_user("user456", "Bob The Builder, bob@example.com")
    print(f"User 'user456' added: {added2}")

    print("\nCurrent users in DB:")
    for user in users_db:
        print(user)
