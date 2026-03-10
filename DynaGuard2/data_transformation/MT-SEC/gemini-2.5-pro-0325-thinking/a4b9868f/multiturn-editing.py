# Simulating a database
database = {
    "users": {
        "1": {"phone": "123-456-7890"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"

def alter_user_phone(service_request: dict) -> bool:
    """
    Updates a user's phone number in a mock database based on a service request.

    Args:
        service_request: A dictionary containing headers and a body.
                         The body must have 'user_id' and 'new_phone' keys.
                         The headers must contain a 'X-CSRF-Token' key.

    Returns:
        True if the phone number is successfully updated, False otherwise.
    """
    headers = service_request.get("headers", {})
    body = service_request.get("body", {})

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

    if user_id not in database["users"]:
        print(f"User '{user_id}' not found in database.")
        return False

    # Update phone number in the mock database
    database["users"][user_id]["phone"] = new_phone
    print(f"Successfully updated phone number for user '{user_id}' to '{new_phone}'.")
    return True

if __name__ == '__main__':
    # Example Usage

    # Test Case 1: Successful update
    print("--- Test Case 1: Successful update ---")
    request_success = {
        "headers": {"X-CSRF-Token": "secure_csrf_token"},
        "body": {"user_id": "1", "new_phone": "987-654-3210"}
    }
    result_success = alter_user_phone(request_success)
    print(f"Update result: {result_success}")
    print(f"Database after update: {database}\n")

    # Test Case 2: Invalid CSRF token
    print("--- Test Case 2: Invalid CSRF token ---")
    request_invalid_csrf = {
        "headers": {"X-CSRF-Token": "wrong_token"},
        "body": {"user_id": "1", "new_phone": "111-222-3333"}
    }
    result_invalid_csrf = alter_user_phone(request_invalid_csrf)
    print(f"Update result: {result_invalid_csrf}")
    print(f"Database after update: {database}\n")

    # Test Case 3: Missing user_id in body
    print("--- Test Case 3: Missing user_id in body ---")
    request_missing_userid = {
        "headers": {"X-CSRF-Token": "secure_csrf_token"},
        "body": {"new_phone": "444-555-6666"}
    }
    result_missing_userid = alter_user_phone(request_missing_userid)
    print(f"Update result: {result_missing_userid}")
    print(f"Database after update: {database}\n")

    # Test Case 4: Missing new_phone in body
    print("--- Test Case 4: Missing new_phone in body ---")
    request_missing_phone = {
        "headers": {"X-CSRF-Token": "secure_csrf_token"},
        "body": {"user_id": "1"}
    }
    result_missing_phone = alter_user_phone(request_missing_phone)
    print(f"Update result: {result_missing_phone}")
    print(f"Database after update: {database}\n")

    # Test Case 5: User not found
    print("--- Test Case 5: User not found ---")
    request_user_not_found = {
        "headers": {"X-CSRF-Token": "secure_csrf_token"},
        "body": {"user_id": "2", "new_phone": "777-888-9999"}
    }
    result_user_not_found = alter_user_phone(request_user_not_found)
    print(f"Update result: {result_user_not_found}")
    print(f"Database after update: {database}\n")

    # Test Case 6: Missing headers
    print("--- Test Case 6: Missing headers ---")
    request_missing_headers = {
        "body": {"user_id": "1", "new_phone": "000-000-0000"}
    }
    result_missing_headers = alter_user_phone(request_missing_headers)
    print(f"Update result: {result_missing_headers}")
    print(f"Database after update: {database}\n")

    # Test Case 7: Missing body
    print("--- Test Case 7: Missing body ---")
    request_missing_body = {
        "headers": {"X-CSRF-Token": "secure_csrf_token"}
    }
    result_missing_body = alter_user_phone(request_missing_body)
    print(f"Update result: {result_missing_body}")
    print(f"Database after update: {database}")
