# Simulating a database
database = {
    "users": {
        "1": {"phone": "123-456-7890"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"

def change_user_phone(request_details: dict) -> bool:
    """
    Updates a user's phone number in the database.

    Ensures a CSRF token is present in the headers and matches the expected token.
    The request_details dictionary should contain 'headers' and 'body'.
    The 'body' should contain 'user_id' and 'new_phone'.

    Args:
        request_details: A dictionary with 'headers' and 'body'.
                         'headers' must contain 'X-CSRF-Token'.
                         'body' must contain 'user_id' and 'new_phone'.

    Returns:
        True if the phone number is successfully updated, False otherwise.
    """
    headers = request_details.get("headers", {})
    body = request_details.get("body", {})

    # Check for CSRF token in headers
    csrf_token = headers.get("X-CSRF-Token")
    if csrf_token != EXPECTED_CSRF_TOKEN:
        print("CSRF token missing or invalid.")
        return False

    # Extract user_id and new_phone from body
    user_id = body.get("user_id")
    new_phone = body.get("new_phone")

    if not user_id or not new_phone:
        print("User ID or new phone number missing in request body.")
        return False

    # Check if user exists in the database
    if user_id in database["users"]:
        # Update the user's phone number
        database["users"][user_id]["phone"] = new_phone
        print(f"Successfully updated phone number for user {user_id}.")
        return True
    else:
        print(f"User {user_id} not found.")
        return False

if __name__ == '__main__':
    # Example Usage

    # Case 1: Successful update
    print("--- Case 1: Successful Update ---")
    request_success = {
        "headers": {"X-CSRF-Token": "secure_csrf_token"},
        "body": {"user_id": "1", "new_phone": "987-654-3210"}
    }
    result_success = change_user_phone(request_success)
    print(f"Update result: {result_success}")
    print(f"Database after successful update: {database}")
    print("-" * 20)

    # Case 2: Invalid CSRF token
    print("--- Case 2: Invalid CSRF Token ---")
    request_invalid_csrf = {
        "headers": {"X-CSRF-Token": "wrong_token"},
        "body": {"user_id": "1", "new_phone": "111-222-3333"}
    }
    result_invalid_csrf = change_user_phone(request_invalid_csrf)
    print(f"Update result: {result_invalid_csrf}")
    print(f"Database after invalid CSRF attempt: {database}") # Phone should be unchanged
    print("-" * 20)

    # Case 3: Missing CSRF token
    print("--- Case 3: Missing CSRF Token ---")
    request_missing_csrf = {
        "headers": {},
        "body": {"user_id": "1", "new_phone": "444-555-6666"}
    }
    result_missing_csrf = change_user_phone(request_missing_csrf)
    print(f"Update result: {result_missing_csrf}")
    print(f"Database after missing CSRF attempt: {database}") # Phone should be unchanged
    print("-" * 20)

    # Case 4: User not found
    print("--- Case 4: User Not Found ---")
    request_user_not_found = {
        "headers": {"X-CSRF-Token": "secure_csrf_token"},
        "body": {"user_id": "2", "new_phone": "777-888-9999"}
    }
    result_user_not_found = change_user_phone(request_user_not_found)
    print(f"Update result: {result_user_not_found}")
    print(f"Database after user not found attempt: {database}")
    print("-" * 20)

    # Case 5: Missing user_id in body
    print("--- Case 5: Missing user_id in body ---")
    request_missing_userid = {
        "headers": {"X-CSRF-Token": "secure_csrf_token"},
        "body": {"new_phone": "123-123-1234"}
    }
    result_missing_userid = change_user_phone(request_missing_userid)
    print(f"Update result: {result_missing_userid}")
    print(f"Database after missing user_id attempt: {database}")
    print("-" * 20)

    # Case 6: Missing new_phone in body
    print("--- Case 6: Missing new_phone in body ---")
    request_missing_phone = {
        "headers": {"X-CSRF-Token": "secure_csrf_token"},
        "body": {"user_id": "1"}
    }
    result_missing_phone = change_user_phone(request_missing_phone)
    print(f"Update result: {result_missing_phone}")
    print(f"Database after missing new_phone attempt: {database}")
    print("-" * 20)
