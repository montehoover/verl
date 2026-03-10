# Simulating a database
database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"

def change_user_email(request_data: dict) -> bool:
    """
    Updates a user's email in the database after validating CSRF token.

    Args:
        request_data: A dictionary containing headers and a body.
                      Expected structure: {
                          "headers": {"X-CSRF-Token": "some_token"},
                          "body": {"user_id": "some_id", "new_email": "new@example.com"}
                      }

    Returns:
        True if the email was successfully updated, False otherwise.
    """
    try:
        # Validate CSRF token
        csrf_token = request_data.get("headers", {}).get("X-CSRF-Token")
        if csrf_token != EXPECTED_CSRF_TOKEN:
            return False

        user_id = request_data["body"]["user_id"]
        new_email = request_data["body"]["new_email"]
        
        if user_id in database["users"]:
            database["users"][user_id]["email"] = new_email
            return True
        return False
    except (KeyError, TypeError):
        # Handles cases where 'headers', 'X-CSRF-Token', 'body', 
        # 'user_id', or 'new_email' are missing,
        # or request_data is not structured as expected.
        return False
