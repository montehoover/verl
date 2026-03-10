import os

# For a real application, this token should be securely generated and managed,
# often per session. For this example, we'll use a fixed value.
# It's also common to store this in an environment variable or configuration file.
# EXPECTED_CSRF_TOKEN = os.environ.get("CSRF_SECRET_TOKEN", "supersecrettoken")
# As per new requirement:
EXPECTED_CSRF_TOKEN = "csrf_token_secured"

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
# As per new requirement:
db_users = {
    "users": {
        "1": {"telephone": "123-556-7890"},
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

def change_user_phone(payload: dict) -> bool:
    """
    Updates a user's phone number securely.

    Validates CSRF token, checks user ID, and updates phone number in db_users.

    Args:
        payload: A dictionary containing:
            - "headers": dict, request headers including "X-CSRF-Token".
            - "user_id": str, the ID of the user.
            - "new_phone_number": str, the new phone number.

    Returns:
        True if the update is successful, False otherwise.
    """
    headers = payload.get("headers")
    if not headers or not isinstance(headers, dict):
        print("Error: 'headers' missing or not a dictionary in payload.")
        return False

    if not check_csrf_token(headers):
        print("CSRF token validation failed.")
        return False

    user_id = payload.get("user_id")
    new_phone_number = payload.get("new_phone_number")

    if not user_id or not isinstance(user_id, str):
        print("Error: 'user_id' missing or invalid in payload.")
        return False
    
    if not new_phone_number or not isinstance(new_phone_number, str):
        print("Error: 'new_phone_number' missing or invalid in payload.")
        return False

    # Check if user exists using get_user_info
    if get_user_info(user_id) is None:
        print(f"User ID '{user_id}' not found.")
        return False

    # Update phone number
    # Ensure the user entry exists before trying to update a sub-key,
    # though get_user_info should have confirmed this.
    if user_id in db_users["users"]:
        db_users["users"][user_id]["telephone"] = new_phone_number
        print(f"Successfully updated phone for user '{user_id}' to '{new_phone_number}'.")
        return True
    else:
        # This case should ideally not be reached if get_user_info works correctly
        print(f"Error: User ID '{user_id}' disappeared before update (should not happen).")
        return False

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
    user_id_found_2 = "2" # This user won't exist with the new db_users
    user_info = get_user_info(user_id_found_2)
    print(f"Info for user '{user_id_found_2}': {user_info}") # Expected: None with new db_users

    print("\n--- Testing change_user_phone ---")
    # Initial state of user "1"
    print(f"Initial info for user '1': {get_user_info('1')}")

    # Case 1: Successful update
    payload_success = {
        "headers": {"X-CSRF-Token": "csrf_token_secured"},
        "user_id": "1",
        "new_phone_number": "555-000-1111"
    }
    success_result = change_user_phone(payload_success)
    print(f"Update success (valid CSRF, user exists): {success_result}") # Expected: True
    print(f"Info for user '1' after update: {get_user_info('1')}") # Expected: {'telephone': '555-000-1111'}

    # Case 2: Invalid CSRF token
    payload_invalid_csrf = {
        "headers": {"X-CSRF-Token": "wrong_csrf_token"},
        "user_id": "1",
        "new_phone_number": "555-222-3333"
    }
    invalid_csrf_result = change_user_phone(payload_invalid_csrf)
    print(f"Update success (invalid CSRF): {invalid_csrf_result}") # Expected: False
    print(f"Info for user '1' (should be unchanged): {get_user_info('1')}")

    # Case 3: User ID not found
    payload_user_not_found = {
        "headers": {"X-CSRF-Token": "csrf_token_secured"},
        "user_id": "nonexistent_user",
        "new_phone_number": "555-444-5555"
    }
    user_not_found_result = change_user_phone(payload_user_not_found)
    print(f"Update success (user not found): {user_not_found_result}") # Expected: False

    # Case 4: Missing 'new_phone_number' in payload
    payload_missing_phone = {
        "headers": {"X-CSRF-Token": "csrf_token_secured"},
        "user_id": "1"
        # "new_phone_number" is missing
    }
    missing_phone_result = change_user_phone(payload_missing_phone)
    print(f"Update success (missing new_phone_number): {missing_phone_result}") # Expected: False

    # Case 5: Missing 'user_id' in payload
    payload_missing_userid = {
        "headers": {"X-CSRF-Token": "csrf_token_secured"},
        "new_phone_number": "555-666-7777"
        # "user_id" is missing
    }
    missing_userid_result = change_user_phone(payload_missing_userid)
    print(f"Update success (missing user_id): {missing_userid_result}") # Expected: False

    # Case 6: Missing 'headers' in payload
    payload_missing_headers = {
        "user_id": "1",
        "new_phone_number": "555-888-9999"
        # "headers" is missing
    }
    missing_headers_result = change_user_phone(payload_missing_headers)
    print(f"Update success (missing headers): {missing_headers_result}") # Expected: False
    print(f"Final info for user '1': {get_user_info('1')}")
