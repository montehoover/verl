# Simulating a database
record_storage = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"


def alter_user_email(payload: dict) -> bool:
    """
    Modify a user's email address using data from a request-like payload.

    Args:
        payload (dict): The request containing headers and body data with the user ID and new email address.

    Returns:
        bool: True if the update is successful, False otherwise.
    """
    try:
        if not isinstance(payload, dict):
            return False

        headers = payload.get("headers") or {}
        body = payload.get("body") or {}

        # CSRF validation
        csrf_token = headers.get("X-CSRF-Token") or headers.get("csrf_token")
        if csrf_token != EXPECTED_CSRF_TOKEN:
            return False

        # Extract and normalize request data
        user_id = body.get("user_id")
        new_email = body.get("email") or body.get("new_email")

        if user_id is None or new_email is None:
            return False

        # Normalize user_id type to string as used in the storage
        user_id = str(user_id)

        # Basic email validation
        if not isinstance(new_email, str) or "@" not in new_email or len(new_email.strip()) == 0:
            return False

        # Check user existence
        users = record_storage.get("users", {})
        if user_id not in users:
            return False

        # Update email
        users[user_id]["email"] = new_email
        return True
    except Exception:
        return False
