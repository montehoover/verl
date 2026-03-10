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

def remove_user_account(input_request):
    """
    Removes a user's account after validating CSRF token.

    Args:
        input_request: A dictionary containing headers with a CSRF token
                       and a body with a user ID.
                       Example: {
                           "headers": {"csrf_token": "secure_csrf_token"},
                           "body": {"user_id": "1"}
                       }

    Returns:
        True if the user account was successfully removed, False otherwise.
    """
    try:
        # Validate CSRF token
        csrf_token = input_request.get("headers", {}).get("csrf_token")
        if csrf_token != EXPECTED_CSRF_TOKEN:
            print("CSRF token validation failed.")
            return False

        # Get user ID
        user_id = input_request.get("body", {}).get("user_id")
        if not user_id:
            print("User ID not provided in request body.")
            return False

        # Check if user exists and remove
        users = database.get("users", {})
        if user_id in users:
            del users[user_id]
            print(f"User {user_id} removed successfully.")
            return True
        else:
            print(f"User {user_id} not found.")
            return False
            
    except (AttributeError, KeyError, TypeError) as e:
        # Handle cases where input_request structure is not as expected
        print(f"Error processing request: {e}")
        return False

if __name__ == '__main__':
    # Example Usage

    # Case 1: Successful removal
    request1 = {
        "headers": {"csrf_token": "secure_csrf_token"},
        "body": {"user_id": "1"}
    }
    result1 = remove_user_account(request1)
    print(f"Removal result for user 1: {result1}") # Expected: True
    print(f"Database after attempting to remove user 1: {database['users']}") # Expected: user 1 removed

    # Case 2: User not found
    request2 = {
        "headers": {"csrf_token": "secure_csrf_token"},
        "body": {"user_id": "5"} # User 5 does not exist
    }
    result2 = remove_user_account(request2)
    print(f"Removal result for user 5: {result2}") # Expected: False
    print(f"Database after attempting to remove non-existent user 5: {database['users']}")

    # Case 3: Invalid CSRF token
    request3 = {
        "headers": {"csrf_token": "invalid_token"},
        "body": {"user_id": "2"}
    }
    result3 = remove_user_account(request3)
    print(f"Removal result for user 2 (invalid CSRF): {result3}") # Expected: False
    print(f"Database after attempting removal with invalid CSRF: {database['users']}") # Expected: user 2 still exists

    # Case 4: Missing CSRF token in headers
    request4 = {
        "headers": {},
        "body": {"user_id": "2"}
    }
    result4 = remove_user_account(request4)
    print(f"Removal result for user 2 (missing CSRF): {result4}") # Expected: False

    # Case 5: Missing user_id in body
    request5 = {
        "headers": {"csrf_token": "secure_csrf_token"},
        "body": {}
    }
    result5 = remove_user_account(request5)
    print(f"Removal result (missing user_id): {result5}") # Expected: False

    # Case 6: Malformed request (e.g., body is not a dict)
    request6 = {
        "headers": {"csrf_token": "secure_csrf_token"},
        "body": "not_a_dict"
    }
    result6 = remove_user_account(request6)
    print(f"Removal result (malformed body): {result6}") # Expected: False
    
    # Case 7: Malformed request (e.g., headers is not a dict)
    request7 = {
        "headers": "not_a_dict",
        "body": {"user_id": "3"}
    }
    result7 = remove_user_account(request7)
    print(f"Removal result (malformed headers): {result7}") # Expected: False
    print(f"Database state: {database['users']}")
