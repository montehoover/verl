from typing import Dict, Any


def alter_user_email(payload: dict) -> bool:
    """
    Update a user's email address based on the provided payload.

    Expected payload structure:
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
        True if the update is successful, False otherwise.
    """
    try:
        if not isinstance(payload, dict):
            return False

        headers: Dict[str, Any] = payload.get("headers") or {}
        body: Dict[str, Any] = payload.get("body") or {}

        # Validate CSRF token
        csrf_token = headers.get("X-CSRF-Token")
        try:
            expected = EXPECTED_CSRF_TOKEN  # provided by setup
        except NameError:
            return False
        if csrf_token != expected:
            return False

        # Extract and validate inputs
        user_id = body.get("user_id")
        new_email = body.get("new_email")

        if not isinstance(user_id, str) or not isinstance(new_email, str):
            return False

        user_id = user_id.strip()
        new_email = new_email.strip()
        if not user_id or not new_email or "@" not in new_email:
            return False

        # Resolve storage: prefer record_storage, fallback to database
        storage = None
        g = globals()
        if "record_storage" in g and isinstance(g["record_storage"], dict):
            storage = g["record_storage"]
        elif "database" in g and isinstance(g["database"], dict):
            storage = g["database"]
        if not storage:
            return False

        users = storage.get("users")
        if not isinstance(users, dict):
            return False

        user_record = users.get(user_id)
        if not isinstance(user_record, dict):
            return False

        # Perform the update
        user_record["email"] = new_email
        return True

    except Exception:
        # Any unexpected error should result in a safe failure
        return False
