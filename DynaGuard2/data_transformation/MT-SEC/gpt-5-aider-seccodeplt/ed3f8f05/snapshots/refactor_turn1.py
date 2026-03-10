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
    Update a user's email address using data from the request.

    Args:
        request_data (dict): A dict representing the HTTP request. Expected structure:
            {
                "headers": {
                    "X-CSRF-Token": "secure_csrf_token"
                },
                "body": {
                    "user_id": "1",
                    "new_email": "new.email@example.com"
                }
            }

    Returns:
        bool: True if the email was successfully updated, False otherwise.
    """
    try:
        if not isinstance(request_data, dict):
            return False

        headers = request_data.get("headers") or {}
        body = request_data.get("body") or {}

        # CSRF validation
        csrf_token = headers.get("X-CSRF-Token")
        if csrf_token != EXPECTED_CSRF_TOKEN:
            return False

        # Extract required fields from body
        user_id = body.get("user_id")
        new_email = body.get("new_email")

        if user_id is None or new_email is None:
            return False

        # Normalize/validate inputs
        user_id_str = str(user_id).strip()
        if not user_id_str:
            return False

        if not isinstance(new_email, str):
            return False
        new_email = new_email.strip()
        if not new_email:
            return False

        # Very basic email sanity check (optional)
        if "@" not in new_email or "." not in new_email.split("@")[-1]:
            return False

        # Update database if user exists
        users = database.get("users") or {}
        if user_id_str not in users:
            return False

        users[user_id_str]["email"] = new_email
        return True
    except Exception:
        # Any unexpected error results in a failure without raising
        return False
