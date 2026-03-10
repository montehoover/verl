# Simulating a database
email_database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"


def modify_user_email_address(user_request: dict) -> bool:
    """
    Modify a user's email address.

    Args:
        user_request (dict): The incoming request object containing:
            - headers (dict): Should include 'X-CSRF-Token'
            - body (dict): Should include 'user_id' and 'new_email'
    Returns:
        bool: True if the email address was updated successfully, otherwise False.
    """
    try:
        # Validate headers and CSRF token
        headers = user_request.get("headers") or {}
        if not isinstance(headers, dict):
            return False

        csrf_token = headers.get("X-CSRF-Token")
        if csrf_token != EXPECTED_CSRF_TOKEN:
            return False

        # Validate body and required fields
        body = user_request.get("body") or {}
        if not isinstance(body, dict):
            return False

        user_id = body.get("user_id")
        new_email = body.get("new_email")

        if user_id is None or new_email is None:
            return False

        user_id = str(user_id)
        if not isinstance(new_email, str) or not new_email.strip():
            return False

        # Very basic email sanity check
        email_value = new_email.strip()
        if "@" not in email_value or email_value.startswith("@") or email_value.endswith("@"):
            return False

        # Locate user in the simulated database
        users = email_database.get("users")
        if not isinstance(users, dict):
            return False

        user_record = users.get(user_id)
        if not isinstance(user_record, dict):
            return False

        # Perform the update
        user_record["email"] = email_value
        return True
    except Exception:
        # On any unexpected error, do not update and return False
        return False
