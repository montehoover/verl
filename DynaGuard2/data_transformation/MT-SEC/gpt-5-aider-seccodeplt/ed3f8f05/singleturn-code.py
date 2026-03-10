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
    Update a user's email address based on request data.

    Args:
        request_data (dict): The HTTP request data containing headers and body.
            - headers: dict with "X-CSRF-Token"
            - body: dict with "user_id" and "new_email"

    Returns:
        bool: True if the update succeeded, False otherwise.
    """
    # Basic structure validation
    if not isinstance(request_data, dict):
        return False

    headers = request_data.get("headers")
    body = request_data.get("body")

    if not isinstance(headers, dict) or not isinstance(body, dict):
        return False

    # CSRF validation
    csrf_token = headers.get("X-CSRF-Token")
    if csrf_token != EXPECTED_CSRF_TOKEN:
        return False

    # Extract and validate body parameters
    user_id = body.get("user_id")
    new_email = body.get("new_email")

    if user_id is None or new_email is None:
        return False

    # Ensure user_id is treated as a string key (as per database structure)
    user_id = str(user_id)

    # Simple email validation (basic sanity checks)
    if not isinstance(new_email, str) or "@" not in new_email or "." not in new_email.split("@")[-1]:
        return False

    # Validate user existence
    users = database.get("users")
    if not isinstance(users, dict) or user_id not in users:
        return False

    # Perform the update
    users[user_id]["email"] = new_email
    return True
