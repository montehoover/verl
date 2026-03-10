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

def remove_user_profile(request_data: dict) -> bool:
    """
    Removes a user's account from the system using their user ID.

    Args:
        request_data: A dictionary containing headers and a body with the user ID.
                      Expected format:
                      {
                          "headers": {"X-CSRF-Token": "some_token"},
                          "body": {"user_id": "some_id"}
                      }

    Returns:
        True if the account was removed successfully, False otherwise.
    """
    headers = request_data.get("headers", {})
    body = request_data.get("body", {})

    # Validate CSRF token
    csrf_token = headers.get("X-CSRF-Token")
    if csrf_token != EXPECTED_CSRF_TOKEN:
        print("CSRF token validation failed.")
        return False

    # Get user ID from body
    user_id = body.get("user_id")
    if not user_id:
        print("User ID not provided in request body.")
        return False

    # Check if user exists and remove
    if user_id in database["users"]:
        del database["users"][user_id]
        print(f"User '{user_id}' removed successfully.")
        return True
    else:
        print(f"User '{user_id}' not found.")
        return False

if __name__ == '__main__':
    # Example Usage

    # Test Case 1: Successful removal
    request1 = {
        "headers": {"X-CSRF-Token": "secure_csrf_token"},
        "body": {"user_id": "1"}
    }
    print("--- Test Case 1: Successful removal ---")
    result1 = remove_user_profile(request1)
    print(f"Result: {result1}") # Expected: True
    print(f"Database after TC1: {database['users']}\n")

    # Test Case 2: Invalid CSRF token
    request2 = {
        "headers": {"X-CSRF-Token": "invalid_token"},
        "body": {"user_id": "2"}
    }
    print("--- Test Case 2: Invalid CSRF token ---")
    result2 = remove_user_profile(request2)
    print(f"Result: {result2}") # Expected: False
    print(f"Database after TC2: {database['users']}\n")

    # Test Case 3: User ID not provided
    request3 = {
        "headers": {"X-CSRF-Token": "secure_csrf_token"},
        "body": {}
    }
    print("--- Test Case 3: User ID not provided ---")
    result3 = remove_user_profile(request3)
    print(f"Result: {result3}") # Expected: False
    print(f"Database after TC3: {database['users']}\n")

    # Test Case 4: User not found
    request4 = {
        "headers": {"X-CSRF-Token": "secure_csrf_token"},
        "body": {"user_id": "non_existent_user"}
    }
    print("--- Test Case 4: User not found ---")
    result4 = remove_user_profile(request4)
    print(f"Result: {result4}") # Expected: False
    print(f"Database after TC4: {database['users']}\n")

    # Test Case 5: Missing headers
    request5 = {
        "body": {"user_id": "2"}
    }
    print("--- Test Case 5: Missing headers ---")
    result5 = remove_user_profile(request5)
    print(f"Result: {result5}") # Expected: False
    print(f"Database after TC5: {database['users']}\n")

    # Test Case 6: Successful removal of another user
    request6 = {
        "headers": {"X-CSRF-Token": "secure_csrf_token"},
        "body": {"user_id": "2"}
    }
    print("--- Test Case 6: Successful removal of another user ---")
    result6 = remove_user_profile(request6)
    print(f"Result: {result6}") # Expected: True
    print(f"Database after TC6: {database['users']}\n")
