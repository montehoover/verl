# Mock database
MOCK_DB = {
    "123": {"email": "alice@example.com", "name": "Alice"},
    "456": {"email": "bob@example.com", "name": "Bob"},
    "789": {"email": "charlie@example.com", "name": "Charlie"},
}

# Predefined authentication token
VALID_AUTH_TOKEN = "secret_token_12345"

def get_user_info(user_id: str, auth_token: str) -> dict | None:
    """
    Retrieves user information from a database if the user is authenticated.

    Args:
        user_id: The ID of the user to retrieve.
        auth_token: The authentication token for the user.

    Returns:
        A dictionary containing the user's email if the user exists and is authenticated,
        or None otherwise.
    """
    if auth_token != VALID_AUTH_TOKEN:
        print("Authentication failed: Invalid token.")
        return None
    
    if user_id in MOCK_DB:
        return {"email": MOCK_DB[user_id]["email"]}
    return None

if __name__ == '__main__':
    # Example Usage
    print("--- Attempting with valid token ---")
    user1_info = get_user_info("123", "secret_token_12345")
    if user1_info:
        print(f"User 123 Email: {user1_info['email']}")
    else:
        print("User 123 not found or authentication failed.")

    user2_info = get_user_info("000", "secret_token_12345")
    if user2_info:
        print(f"User 000 Email: {user2_info['email']}")
    else:
        print("User 000 not found or authentication failed.")

    user3_info = get_user_info("456", "secret_token_12345")
    if user3_info:
        print(f"User 456 Email: {user3_info['email']}")
    else:
        print("User 456 not found or authentication failed.")

    print("\n--- Attempting with invalid token ---")
    user4_info = get_user_info("123", "wrong_token")
    if user4_info:
        print(f"User 123 Email: {user4_info['email']}")
    else:
        print("User 123 not found or authentication failed.")
