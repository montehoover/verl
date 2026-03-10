import os

# For a real application, this token should be securely generated and managed,
# often per session. For this example, we'll use a fixed value.
# It's also common to store this in an environment variable or configuration file.
EXPECTED_CSRF_TOKEN = os.environ.get("CSRF_SECRET_TOKEN", "supersecrettoken")

def check_csrf_token(headers: dict) -> bool:
    """
    Verifies if the provided CSRF token in headers matches an expected token.

    Args:
        headers: A dictionary representing the incoming request headers.
                 Expected to contain a key like 'X-CSRF-Token'.

    Returns:
        True if the token is valid, False otherwise.
    """
    provided_token = headers.get("X-CSRF-Token")
    if provided_token and provided_token == EXPECTED_CSRF_TOKEN:
        return True
    return False

# Simulating a database
db_users = {
    "users": {
        "1": {"telephone": "123-556-7890"},
        "2": {"telephone": "987-654-3210", "username": "user2"},
    }
}

def get_user_info(user_id: str) -> dict | None:
    """
    Retrieves user information from the simulated database.

    Args:
        user_id: The ID of the user to retrieve.

    Returns:
        A dictionary containing user data if found, otherwise None.
    """
    return db_users["users"].get(user_id)

if __name__ == '__main__':
    # Example Usage for check_csrf_token:

    # Case 1: Valid token
    valid_headers = {"X-CSRF-Token": "supersecrettoken", "Content-Type": "application/json"}
    is_valid = check_csrf_token(valid_headers)
    print(f"Request with valid token is valid: {is_valid}") # Expected: True

    # Case 2: Invalid token
    invalid_headers = {"X-CSRF-Token": "wrongtoken", "Content-Type": "application/json"}
    is_valid = check_csrf_token(invalid_headers)
    print(f"Request with invalid token is valid: {is_valid}") # Expected: False

    # Case 3: Missing token
    missing_token_headers = {"Content-Type": "application/json"}
    is_valid = check_csrf_token(missing_token_headers)
    print(f"Request with missing token is valid: {is_valid}") # Expected: False

    # Case 4: Different header name (if you expect a different one)
    # For example, if the token could also be in 'HTTP_X_CSRFTOKEN' (like in Django)
    # You might need to adjust the .get() or add more checks.
    # For this example, we stick to 'X-CSRF-Token'.

    # To run this example with an environment variable:
    # CSRF_SECRET_TOKEN="anothersecret" python multiturn-editing.py
    # This will change the EXPECTED_CSRF_TOKEN if the env var is set.
    # If CSRF_SECRET_TOKEN is set to "anothersecret":
    #   valid_headers with "supersecrettoken" will become invalid.
    #   You'd need valid_headers = {"X-CSRF-Token": "anothersecret", ...}
    print(f"Expected CSRF token is: {EXPECTED_CSRF_TOKEN}")

    print("\n--- Testing get_user_info ---")
    # Example Usage for get_user_info:

    # Case 1: User found
    user_id_found = "1"
    user_info = get_user_info(user_id_found)
    print(f"Info for user '{user_id_found}': {user_info}") # Expected: {'telephone': '123-556-7890'}

    # Case 2: User not found
    user_id_not_found = "3"
    user_info = get_user_info(user_id_not_found)
    print(f"Info for user '{user_id_not_found}': {user_info}") # Expected: None

    # Case 3: Another user found
    user_id_found_2 = "2"
    user_info = get_user_info(user_id_found_2)
    print(f"Info for user '{user_id_found_2}': {user_info}") # Expected: {'telephone': '987-654-3210', 'username': 'user2'}
