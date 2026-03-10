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

def terminate_user_account(request_data: dict) -> bool:
    """
    Facilitates the deletion of a user's account from a mock database.

    Args:
        request_data: A dictionary representing an HTTP request with 'headers'
                      for the CSRF token and 'body' containing the user ID.

    Returns:
        True if the user's account is successfully deleted, otherwise False.
    """
    headers = request_data.get("headers", {})
    body = request_data.get("body", {})

    csrf_token = headers.get("X-CSRF-Token")
    user_id = body.get("user_id")

    if csrf_token != EXPECTED_CSRF_TOKEN:
        print("CSRF token validation failed.")
        return False

    if not user_id:
        print("User ID not provided in request body.")
        return False

    if user_id in database["users"]:
        del database["users"][user_id]
        print(f"User account {user_id} successfully terminated.")
        return True
    else:
        print(f"User ID {user_id} not found in database.")
        return False

if __name__ == '__main__':
    # Test cases
    print("Initial database state:", database["users"])

    # Test case 1: Successful deletion
    request_success = {
        "headers": {"X-CSRF-Token": "secure_csrf_token"},
        "body": {"user_id": "1"}
    }
    print("\nAttempting to delete user '1' (should succeed):")
    result_success = terminate_user_account(request_success)
    print(f"Deletion successful: {result_success}")
    print("Database state after attempt:", database["users"])

    # Test case 2: Invalid CSRF token
    request_invalid_csrf = {
        "headers": {"X-CSRF-Token": "invalid_token"},
        "body": {"user_id": "2"}
    }
    print("\nAttempting to delete user '2' with invalid CSRF (should fail):")
    result_invalid_csrf = terminate_user_account(request_invalid_csrf)
    print(f"Deletion successful: {result_invalid_csrf}")
    print("Database state after attempt:", database["users"])

    # Test case 3: User ID not found
    request_user_not_found = {
        "headers": {"X-CSRF-Token": "secure_csrf_token"},
        "body": {"user_id": "99"}
    }
    print("\nAttempting to delete user '99' (should fail - user not found):")
    result_user_not_found = terminate_user_account(request_user_not_found)
    print(f"Deletion successful: {result_user_not_found}")
    print("Database state after attempt:", database["users"])

    # Test case 4: Missing CSRF token in headers
    request_missing_csrf = {
        "headers": {},
        "body": {"user_id": "3"}
    }
    print("\nAttempting to delete user '3' with missing CSRF (should fail):")
    result_missing_csrf = terminate_user_account(request_missing_csrf)
    print(f"Deletion successful: {result_missing_csrf}")
    print("Database state after attempt:", database["users"])

    # Test case 5: Missing user_id in body
    request_missing_userid = {
        "headers": {"X-CSRF-Token": "secure_csrf_token"},
        "body": {}
    }
    print("\nAttempting to delete with missing user_id (should fail):")
    result_missing_userid = terminate_user_account(request_missing_userid)
    print(f"Deletion successful: {result_missing_userid}")
    print("Database state after attempt:", database["users"])

    # Test case 6: Successful deletion of another user
    request_success_2 = {
        "headers": {"X-CSRF-Token": "secure_csrf_token"},
        "body": {"user_id": "4"}
    }
    print("\nAttempting to delete user '4' (should succeed):")
    result_success_2 = terminate_user_account(request_success_2)
    print(f"Deletion successful: {result_success_2}")
    print("Database state after attempt:", database["users"])
