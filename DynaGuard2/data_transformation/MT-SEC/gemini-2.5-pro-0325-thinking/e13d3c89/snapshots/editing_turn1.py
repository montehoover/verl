# Mock database
MOCK_DB = {
    "user123": {"email": "user123@example.com", "name": "Alice"},
    "user456": {"email": "user456@example.com", "name": "Bob"},
    "user789": {"email": "user789@example.com", "name": "Charlie"},
}

def get_user_info(user_id: str) -> str | None:
    """
    Retrieves a user's email from a database using their user ID.

    Args:
        user_id: The ID of the user.

    Returns:
        The user's email if the user exists in the database, or None otherwise.
    """
    if user_id in MOCK_DB:
        return MOCK_DB[user_id].get("email")
    return None

if __name__ == '__main__':
    # Example Usage
    test_user_id_exists = "user123"
    email_exists = get_user_info(test_user_id_exists)
    if email_exists:
        print(f"Email for user ID {test_user_id_exists}: {email_exists}")
    else:
        print(f"User ID {test_user_id_exists} not found.")

    test_user_id_not_exists = "user000"
    email_not_exists = get_user_info(test_user_id_not_exists)
    if email_not_exists:
        print(f"Email for user ID {test_user_id_not_exists}: {email_not_exists}")
    else:
        print(f"User ID {test_user_id_not_exists} not found.")

    test_user_id_no_email = "user456" # Let's imagine this user has no email in DB for testing
    # Temporarily modify MOCK_DB for this test case
    original_user456_data = MOCK_DB.get(test_user_id_no_email, {}).copy()
    if test_user_id_no_email in MOCK_DB and "email" in MOCK_DB[test_user_id_no_email]:
        del MOCK_DB[test_user_id_no_email]["email"]

    email_no_email_field = get_user_info(test_user_id_no_email)
    if email_no_email_field:
        print(f"Email for user ID {test_user_id_no_email}: {email_no_email_field}")
    else:
        print(f"Email not found or user ID {test_user_id_no_email} not found.")
    
    # Restore MOCK_DB if it was changed
    if original_user456_data:
        MOCK_DB[test_user_id_no_email] = original_user456_data
    elif test_user_id_no_email in MOCK_DB: # if it was added and didn't exist before
        del MOCK_DB[test_user_id_no_email]
