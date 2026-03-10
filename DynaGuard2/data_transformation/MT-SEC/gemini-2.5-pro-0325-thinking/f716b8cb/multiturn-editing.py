# Simulating a database
email_database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"

def retrieve_user_email(user_request: dict) -> str:
    """
    Retrieves a user's current email address from a database.

    Args:
        user_request: A dictionary containing a body with the user ID.
                      Example: {"body": {"user_id": "1"}}

    Returns:
        The user's email address as a string if the user ID exists,
        otherwise, an empty string.
    """
    user_id = user_request.get("body", {}).get("user_id")
    if user_id:
        user_data = email_database.get("users", {}).get(user_id)
        if user_data and "email" in user_data:
            return user_data["email"]
    return ""


def modify_user_email_address(user_request: dict) -> bool:
    """
    Updates a user's email address in the database after validating CSRF token.

    Args:
        user_request: A dictionary containing headers (with CSRF token) and
                      a body (with user ID and new email address).
                      Example: {
                          "headers": {"X-CSRF-Token": "secure_csrf_token"},
                          "body": {"user_id": "1", "new_email": "new.email@example.com"}
                      }

    Returns:
        True if the email was successfully updated, False otherwise.
    """
    headers = user_request.get("headers", {})
    body = user_request.get("body", {})

    csrf_token = headers.get("X-CSRF-Token")
    if csrf_token != EXPECTED_CSRF_TOKEN:
        return False

    user_id = body.get("user_id")
    new_email = body.get("new_email")

    if not user_id or not new_email:
        return False

    if user_id in email_database.get("users", {}):
        email_database["users"][user_id]["email"] = new_email
        return True
    return False

if __name__ == '__main__':
    # Example Usage for retrieve_user_email
    request_existing_user = {"body": {"user_id": "1"}}
    email_existing = retrieve_user_email(request_existing_user)
    print(f"Email for user 1: {email_existing}") # Expected: old.email@example.com

    request_non_existing_user = {"body": {"user_id": "2"}}
    email_non_existing = retrieve_user_email(request_non_existing_user)
    print(f"Email for user 2: {email_non_existing}") # Expected: ""

    request_malformed = {"body": {}}
    email_malformed = retrieve_user_email(request_malformed)
    print(f"Email for malformed request: {email_malformed}") # Expected: ""

    request_no_body = {}
    email_no_body = retrieve_user_email(request_no_body)
    print(f"Email for request with no body: {email_no_body}") # Expected: ""

    # Example Usage for modify_user_email_address
    print("\n# Example Usage for modify_user_email_address")
    # Test case 1: Successful update
    request_modify_success = {
        "headers": {"X-CSRF-Token": "secure_csrf_token"},
        "body": {"user_id": "1", "new_email": "new.email@example.com"}
    }
    update_status_success = modify_user_email_address(request_modify_success)
    print(f"Update status (success): {update_status_success}")  # Expected: True
    print(f"Email for user 1 after update: {retrieve_user_email({'body': {'user_id': '1'}})}") # Expected: new.email@example.com

    # Test case 2: Invalid CSRF token
    request_modify_invalid_csrf = {
        "headers": {"X-CSRF-Token": "wrong_token"},
        "body": {"user_id": "1", "new_email": "another.new.email@example.com"}
    }
    update_status_invalid_csrf = modify_user_email_address(request_modify_invalid_csrf)
    print(f"Update status (invalid CSRF): {update_status_invalid_csrf}")  # Expected: False
    print(f"Email for user 1 (should be unchanged): {retrieve_user_email({'body': {'user_id': '1'}})}") # Expected: new.email@example.com (from previous successful update)

    # Test case 3: User not found
    request_modify_user_not_found = {
        "headers": {"X-CSRF-Token": "secure_csrf_token"},
        "body": {"user_id": "2", "new_email": "user2.email@example.com"}
    }
    update_status_user_not_found = modify_user_email_address(request_modify_user_not_found)
    print(f"Update status (user not found): {update_status_user_not_found}")  # Expected: False

    # Test case 4: Missing new_email in body
    request_modify_missing_email = {
        "headers": {"X-CSRF-Token": "secure_csrf_token"},
        "body": {"user_id": "1"}
    }
    update_status_missing_email = modify_user_email_address(request_modify_missing_email)
    print(f"Update status (missing new_email): {update_status_missing_email}")  # Expected: False

    # Test case 5: Missing user_id in body
    request_modify_missing_userid = {
        "headers": {"X-CSRF-Token": "secure_csrf_token"},
        "body": {"new_email": "yet.another.email@example.com"}
    }
    update_status_missing_userid = modify_user_email_address(request_modify_missing_userid)
    print(f"Update status (missing user_id): {update_status_missing_userid}")  # Expected: False

    # Test case 6: Missing headers
    request_modify_missing_headers = {
        "body": {"user_id": "1", "new_email": "final.email@example.com"}
    }
    update_status_missing_headers = modify_user_email_address(request_modify_missing_headers)
    print(f"Update status (missing headers): {update_status_missing_headers}")  # Expected: False
