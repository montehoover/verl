# Simulating a database
database = {
    "users": {
        "1": {"email": "user1@example.com"},
        "2": {"email": "user2@example.com"},
        "3": {"email": "user3@example.com"},
        "4": {"email": "user4@example.com"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"

def remove_account(request: dict) -> bool:
    """
    Removes a user account from the system.

    Args:
        request: A dictionary representing an HTTP request containing headers
                 and a body with the user ID.
                 Example:
                 {
                     "headers": {"X-CSRF-Token": "some_token"},
                     "body": {"user_id": "1"}
                 }

    Returns:
        True if the user account is successfully deleted, False otherwise.
    """
    headers = request.get("headers", {})
    body = request.get("body", {})

    # Check for CSRF token in headers
    csrf_token = headers.get("X-CSRF-Token")
    if csrf_token != EXPECTED_CSRF_TOKEN:
        print("CSRF token validation failed.")
        return False

    user_id = body.get("user_id")
    if not user_id:
        print("User ID not provided in request body.")
        return False

    if user_id in database["users"]:
        del database["users"][user_id]
        print(f"User account {user_id} removed successfully.")
        return True
    else:
        print(f"User account {user_id} not found.")
        return False

if __name__ == '__main__':
    # Example Usage
    print("Initial database state:", database["users"])

    # Scenario 1: Successful deletion
    request_valid = {
        "headers": {"X-CSRF-Token": "secure_csrf_token"},
        "body": {"user_id": "1"}
    }
    print("\nAttempting to remove user '1' (valid CSRF):")
    result1 = remove_account(request_valid)
    print(f"Deletion successful: {result1}")
    print("Database state after attempt 1:", database["users"])

    # Scenario 2: Invalid CSRF token
    request_invalid_csrf = {
        "headers": {"X-CSRF-Token": "wrong_token"},
        "body": {"user_id": "2"}
    }
    print("\nAttempting to remove user '2' (invalid CSRF):")
    result2 = remove_account(request_invalid_csrf)
    print(f"Deletion successful: {result2}")
    print("Database state after attempt 2:", database["users"])

    # Scenario 3: User not found
    request_user_not_found = {
        "headers": {"X-CSRF-Token": "secure_csrf_token"},
        "body": {"user_id": "999"}
    }
    print("\nAttempting to remove user '999' (user not found):")
    result3 = remove_account(request_user_not_found)
    print(f"Deletion successful: {result3}")
    print("Database state after attempt 3:", database["users"])

    # Scenario 4: Missing user_id in body
    request_missing_userid = {
        "headers": {"X-CSRF-Token": "secure_csrf_token"},
        "body": {}
    }
    print("\nAttempting to remove user (missing user_id):")
    result4 = remove_account(request_missing_userid)
    print(f"Deletion successful: {result4}")
    print("Database state after attempt 4:", database["users"])

    # Scenario 5: Missing headers
    request_missing_headers = {
        "body": {"user_id": "3"}
    }
    print("\nAttempting to remove user '3' (missing headers):")
    result5 = remove_account(request_missing_headers)
    print(f"Deletion successful: {result5}")
    print("Database state after attempt 5:", database["users"])
