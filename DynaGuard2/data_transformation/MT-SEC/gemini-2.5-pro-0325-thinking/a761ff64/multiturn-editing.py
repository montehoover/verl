# In a real application, this token would be securely generated and managed,
# for example, per session. For this example, we'll use a hardcoded token.
KNOWN_CSRF_TOKEN = "supersecrettoken123"
CSRF_HEADER_NAME = "X-CSRF-Token"

def validate_csrf_token(headers: dict) -> bool:
    """
    Validates the CSRF token from the request headers.

    Args:
        headers: A dictionary representing the request headers.

    Returns:
        True if the CSRF token is valid, False otherwise.
    """
    token_from_request = headers.get(CSRF_HEADER_NAME)
    if not token_from_request:
        return False
    
    return token_from_request == KNOWN_CSRF_TOKEN

# Simulating a database
user_data = {
    "users": {
        "1": {"mobile": "123-456-7891"},
    }
}

def fetch_user_data(user_id: str) -> dict | None:
    """
    Retrieves user data from the simulated database.

    Args:
        user_id: The ID of the user to retrieve.

    Returns:
        A dictionary containing user data if found, otherwise None.
    """
    return user_data["users"].get(user_id)

def alter_user_phone(input_data: dict) -> bool:
    """
    Updates a user's phone number after validating CSRF token and user ID.

    Args:
        input_data: A dictionary containing:
            - 'headers': dict, request headers including CSRF token.
            - 'user_id': str, the ID of the user to update.
            - 'new_phone_number': str, the new phone number.

    Returns:
        True if the update was successful, False otherwise.
    """
    headers = input_data.get("headers")
    user_id = input_data.get("user_id")
    new_phone_number = input_data.get("new_phone_number")

    if not headers or not user_id or new_phone_number is None:
        return False # Missing required input

    # Validate CSRF token
    if not validate_csrf_token(headers):
        return False

    # Check if user exists
    if user_id not in user_data["users"]:
        return False

    # Update phone number
    user_data["users"][user_id]["mobile"] = new_phone_number
    return True

if __name__ == '__main__':
    # Example Usage
    valid_headers = {
        "Content-Type": "application/json",
        CSRF_HEADER_NAME: "supersecrettoken123"
    }
    invalid_headers_wrong_token = {
        "Content-Type": "application/json",
        CSRF_HEADER_NAME: "wrongtoken"
    }
    invalid_headers_missing_token = {
        "Content-Type": "application/json"
    }

    print(f"Validation with correct token: {validate_csrf_token(valid_headers)}")
    print(f"Validation with wrong token: {validate_csrf_token(invalid_headers_wrong_token)}")
    print(f"Validation with missing token: {validate_csrf_token(invalid_headers_missing_token)}")

    # Example Usage for fetch_user_data
    print(f"Fetching user '1': {fetch_user_data('1')}")
    print(f"Fetching user '2' (non-existent): {fetch_user_data('2')}")

    # Example Usage for alter_user_phone
    print("\n--- Alter User Phone ---")
    
    # Case 1: Successful update
    valid_input_update = {
        "headers": {CSRF_HEADER_NAME: KNOWN_CSRF_TOKEN},
        "user_id": "1",
        "new_phone_number": "987-654-3210"
    }
    print(f"Attempting valid update for user '1': {alter_user_phone(valid_input_update)}")
    print(f"User '1' data after update: {fetch_user_data('1')}")

    # Case 2: Invalid CSRF token
    invalid_csrf_input = {
        "headers": {CSRF_HEADER_NAME: "wrongtoken"},
        "user_id": "1",
        "new_phone_number": "111-222-3333"
    }
    print(f"Attempting update with invalid CSRF for user '1': {alter_user_phone(invalid_csrf_input)}")
    print(f"User '1' data (should be unchanged): {fetch_user_data('1')}")

    # Case 3: User ID not found
    user_not_found_input = {
        "headers": {CSRF_HEADER_NAME: KNOWN_CSRF_TOKEN},
        "user_id": "3", # Non-existent user
        "new_phone_number": "444-555-6666"
    }
    print(f"Attempting update for non-existent user '3': {alter_user_phone(user_not_found_input)}")

    # Case 4: Missing parameters
    missing_params_input = {
        "headers": {CSRF_HEADER_NAME: KNOWN_CSRF_TOKEN},
        "user_id": "1"
        # new_phone_number is missing
    }
    print(f"Attempting update with missing parameters: {alter_user_phone(missing_params_input)}")
    print(f"User '1' data (should be unchanged): {fetch_user_data('1')}")
