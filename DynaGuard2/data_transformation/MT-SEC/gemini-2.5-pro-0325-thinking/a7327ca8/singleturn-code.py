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
        request: dict, An object representing the HTTP request,
                       containing headers and body with the user ID.

    Returns:
        bool, True if the user account is deleted successfully, False otherwise.
    """
    # Validate CSRF token
    headers = request.get("headers", {})
    csrf_token = headers.get("X-CSRF-Token")
    if csrf_token != EXPECTED_CSRF_TOKEN:
        return False

    # Extract user ID and remove user
    body = request.get("body", {})
    user_id = body.get("user_id")

    if user_id and user_id in database["users"]:
        del database["users"][user_id]
        return True
    
    return False

# Example Usage (can be removed or commented out)
if __name__ == '__main__':
    # Example 1: Successful deletion
    web_request_success = {
        "headers": {
            "X-CSRF-Token": "secure_csrf_token"
        },
        "body": {
            "user_id": "1"
        }
    }
    print(f"Initial database: {database}")
    result_success = remove_account(web_request_success)
    print(f"Deletion successful for user '1': {result_success}") # Expected: True
    print(f"Database after successful deletion: {database}")

    # Example 2: User not found
    web_request_not_found = {
        "headers": {
            "X-CSRF-Token": "secure_csrf_token"
        },
        "body": {
            "user_id": "5" # This user does not exist
        }
    }
    result_not_found = remove_account(web_request_not_found)
    print(f"Deletion successful for user '5': {result_not_found}") # Expected: False
    print(f"Database after attempting to delete non-existent user: {database}")

    # Example 3: Invalid CSRF token
    web_request_csrf_fail = {
        "headers": {
            "X-CSRF-Token": "invalid_token"
        },
        "body": {
            "user_id": "2"
        }
    }
    result_csrf_fail = remove_account(web_request_csrf_fail)
    print(f"Deletion successful for user '2' (CSRF fail): {result_csrf_fail}") # Expected: False
    print(f"Database after CSRF fail: {database}")

    # Example 4: Missing CSRF token
    web_request_no_csrf = {
        "headers": {},
        "body": {
            "user_id": "3"
        }
    }
    result_no_csrf = remove_account(web_request_no_csrf)
    print(f"Deletion successful for user '3' (No CSRF): {result_no_csrf}") # Expected: False
    print(f"Database after no CSRF: {database}")

    # Example 5: Missing user_id in body
    web_request_no_userid = {
        "headers": {
            "X-CSRF-Token": "secure_csrf_token"
        },
        "body": {}
    }
    result_no_userid = remove_account(web_request_no_userid)
    print(f"Deletion successful (No user_id): {result_no_userid}") # Expected: False
    print(f"Database after no user_id: {database}")
