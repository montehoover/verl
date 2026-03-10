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
    Modify a user's email address based on an incoming request.

    Args:
        user_request (dict): Incoming request with:
            - headers: dict, must include "X-CSRF-Token"
            - body: dict, must include "user_id" (str) and "new_email" (str)

    Returns:
        bool: True if the update was successful, False otherwise.
    """
    if not isinstance(user_request, dict):
        return False

    # CSRF validation
    headers = user_request.get("headers")
    if not isinstance(headers, dict):
        return False
    csrf_token = headers.get("X-CSRF-Token")
    if csrf_token != EXPECTED_CSRF_TOKEN:
        return False

    # Extract and validate body
    body = user_request.get("body")
    if not isinstance(body, dict):
        return False

    user_id = body.get("user_id")
    new_email = body.get("new_email")

    if not isinstance(user_id, str) or not isinstance(new_email, str):
        return False

    user_id = user_id.strip()
    new_email = new_email.strip()

    if not user_id or not new_email or "@" not in new_email:
        return False

    # Locate user and update email
    users = email_database.get("users")
    if not isinstance(users, dict) or user_id not in users:
        return False

    user_record = users.get(user_id)
    if not isinstance(user_record, dict):
        return False

    user_record["email"] = new_email
    return True
