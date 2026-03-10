# Mock database
MOCK_DB = {
    "user123": {"email": "user123@example.com", "name": "Alice"},
    "user456": {"email": "user456@example.com", "name": "Bob"},
    "user789": {"email": "user789@example.com", "name": "Charlie"},
}

# Mock CSRF token store or validation mechanism
VALID_CSRF_TOKEN = "supersecrettoken"

def is_csrf_token_valid(token: str) -> bool:
    """
    Validates the CSRF token.
    In a real application, this would involve checking against a session token or a more robust mechanism.
    """
    return token == VALID_CSRF_TOKEN

def get_user_info(user_id: str, csrf_token: str) -> str | None:
    """
    Retrieves a user's email from a database using their user ID, after validating CSRF token.

    Args:
        user_id: The ID of the user.
        csrf_token: The CSRF token for request validation.

    Returns:
        The user's email if the user exists in the database and CSRF token is valid, or None otherwise.
    """
    if not is_csrf_token_valid(csrf_token):
        print("CSRF token is invalid.")
        return None
    if user_id in MOCK_DB:
        return MOCK_DB[user_id].get("email")
    return None

if __name__ == '__main__':
    # Example Usage
    test_user_id_exists = "user123"
    valid_token = "supersecrettoken"
    invalid_token = "wrongtoken"

    # Test case 1: User exists, valid token
    email_exists = get_user_info(test_user_id_exists, valid_token)
    if email_exists:
        print(f"Email for user ID {test_user_id_exists} (valid token): {email_exists}")
    else:
        print(f"User ID {test_user_id_exists} not found or token invalid.")

    # Test case 2: User exists, invalid token
    email_invalid_token = get_user_info(test_user_id_exists, invalid_token)
    if email_invalid_token:
        print(f"Email for user ID {test_user_id_exists} (invalid token): {email_invalid_token}")
    else:
        print(f"User ID {test_user_id_exists} not found or token invalid.")

    # Test case 3: User does not exist, valid token
    test_user_id_not_exists = "user000"
    email_not_exists = get_user_info(test_user_id_not_exists, valid_token)
    if email_not_exists:
        print(f"Email for user ID {test_user_id_not_exists} (valid token): {email_not_exists}")
    else:
        print(f"User ID {test_user_id_not_exists} not found or token invalid.")

    # Test case 4: User has no email field, valid token
    test_user_id_no_email = "user456" # Let's imagine this user has no email in DB for testing
    # Temporarily modify MOCK_DB for this test case
    original_user456_data = MOCK_DB.get(test_user_id_no_email, {}).copy()
    if test_user_id_no_email in MOCK_DB and "email" in MOCK_DB[test_user_id_no_email]:
        del MOCK_DB[test_user_id_no_email]["email"]

    email_no_email_field = get_user_info(test_user_id_no_email, valid_token)
    if email_no_email_field:
        print(f"Email for user ID {test_user_id_no_email} (valid token): {email_no_email_field}")
    else:
        print(f"Email not found or user ID {test_user_id_no_email} not found or token invalid.")
    
    # Restore MOCK_DB if it was changed
    if original_user456_data: # if it existed before and was modified
        MOCK_DB[test_user_id_no_email] = original_user456_data
    elif test_user_id_no_email in MOCK_DB and not original_user456_data : # if it was added and didn't exist before
        # This case should not happen with current MOCK_DB setup, but good for robustness
        del MOCK_DB[test_user_id_no_email]
    # If original_user456_data is empty and test_user_id_no_email is not in MOCK_DB, nothing to do.
