# Define a predefined CSRF token (in a real application, this should be securely managed)
PREDEFINED_CSRF_TOKEN = "supersecrettoken123"

def check_user_existence(user_id: str, headers: dict) -> bool:
    """
    Verifies if a user exists in a system's database and validates a CSRF token.

    Args:
        user_id: The ID of the user to check.
        headers: A dictionary containing request headers, expected to include 'X-CSRF-Token'.

    Returns:
        True if the user exists and the CSRF token is valid, False otherwise.
    """
    # This is a placeholder for actual database lookup logic.
    # In a real application, you would query your database here.
    # For example:
    # db_users = {"user123": {"name": "Alice"}, "user456": {"name": "Bob"}}
    # user_exists = user_id in db_users
    
    # Placeholder user existence check
    print(f"Checking existence of user: {user_id}")
    user_exists = user_id == "existing_user"

    # CSRF token validation
    csrf_token_from_header = headers.get("X-CSRF-Token")
    print(f"CSRF token from header: {csrf_token_from_header}")
    is_csrf_valid = csrf_token_from_header == PREDEFINED_CSRF_TOKEN

    if not user_exists:
        print(f"User '{user_id}' does not exist.")
        return False
    
    if not is_csrf_valid:
        print("CSRF token is invalid.")
        return False

    print(f"User '{user_id}' exists and CSRF token is valid.")
    return True

if __name__ == '__main__':
    # Example Usage
    user1 = "existing_user"
    user2 = "non_existing_user"

    headers_valid_token = {"X-CSRF-Token": "supersecrettoken123"}
    headers_invalid_token = {"X-CSRF-Token": "wrongtoken"}
    headers_no_token = {}

    print("--- Test Case 1: Existing user, valid token ---")
    print(f"Result: {check_user_existence(user1, headers_valid_token)}\n") # Expected: True

    print("--- Test Case 2: Existing user, invalid token ---")
    print(f"Result: {check_user_existence(user1, headers_invalid_token)}\n") # Expected: False

    print("--- Test Case 3: Existing user, no token ---")
    print(f"Result: {check_user_existence(user1, headers_no_token)}\n") # Expected: False

    print("--- Test Case 4: Non-existing user, valid token ---")
    print(f"Result: {check_user_existence(user2, headers_valid_token)}\n") # Expected: False
    
    print("--- Test Case 5: Non-existing user, invalid token ---")
    print(f"Result: {check_user_existence(user2, headers_invalid_token)}\n") # Expected: False
