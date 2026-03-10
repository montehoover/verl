# Simulating a database
database = {
    "users": {
        "1": {"phone": "123-456-7890"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"

def modify_user_phone(user_request: dict) -> bool:
    """
    Updates a user's phone number in the database after validating CSRF token.

    Args:
        user_request: A dictionary containing headers (with CSRF token)
                      and body (with user_id and new_phone).

    Returns:
        True if the update is successful, False otherwise.
    """
    headers = user_request.get("headers", {})
    body = user_request.get("body", {})

    # Validate CSRF token
    csrf_token = headers.get("X-CSRF-Token")
    if csrf_token != EXPECTED_CSRF_TOKEN:
        print("CSRF token validation failed.")
        return False

    user_id = body.get("user_id")
    new_phone = body.get("new_phone")

    if not user_id or not new_phone:
        print("User ID or new phone number missing in request body.")
        return False

    if user_id in database["users"]:
        database["users"][user_id]["phone"] = new_phone
        print(f"User '{user_id}' phone number updated to '{new_phone}'.")
        return True
    else:
        print(f"User '{user_id}' not found.")
        return False

if __name__ == '__main__':
    # Example usage:
    
    # Scenario 1: Successful update
    request_valid = {
        "headers": {"X-CSRF-Token": "secure_csrf_token"},
        "body": {"user_id": "1", "new_phone": "987-654-3210"}
    }
    if modify_user_phone(request_valid):
        print(f"Database after successful update: {database}")
    else:
        print("Update failed.")

    print("-" * 20)

    # Scenario 2: Invalid CSRF token
    request_invalid_csrf = {
        "headers": {"X-CSRF-Token": "wrong_token"},
        "body": {"user_id": "1", "new_phone": "111-222-3333"}
    }
    if modify_user_phone(request_invalid_csrf):
        print("This should not happen.")
    else:
        print(f"Database after failed CSRF attempt: {database}")
        
    print("-" * 20)

    # Scenario 3: User not found
    request_user_not_found = {
        "headers": {"X-CSRF-Token": "secure_csrf_token"},
        "body": {"user_id": "2", "new_phone": "555-555-5555"}
    }
    if modify_user_phone(request_user_not_found):
        print("This should not happen.")
    else:
        print(f"Database after user not found attempt: {database}")

    print("-" * 20)
    
    # Scenario 4: Missing user_id in body
    request_missing_userid = {
        "headers": {"X-CSRF-Token": "secure_csrf_token"},
        "body": {"new_phone": "555-555-5555"}
    }
    if modify_user_phone(request_missing_userid):
        print("This should not happen.")
    else:
        print(f"Database after missing user_id attempt: {database}")
