from typing import Dict, Any

def change_user_email(new_request: dict) -> bool:
    """
    Modify a user's email address using data from an incoming request.

    Args:
        new_request (dict): {
            "headers": {"X-CSRF-Token": str},
            "body": {"user_id": str, "new_email": str}
        }

    Returns:
        bool: True if the email address was updated successfully, else False.
    """
    if not isinstance(new_request, dict):
        return False

    headers = new_request.get("headers") or {}
    body = new_request.get("body") or {}

    # Validate CSRF token against expected global
    token = headers.get("X-CSRF-Token")
    expected_token = globals().get("VALID_CSRF_TOKEN")
    if expected_token is None:
        # Support alternate global name mentioned in context
        expected_token = globals().get("EXPECTED_CSRF_TOKEN")

    if not expected_token or token != expected_token:
        return False

    # Extract and validate inputs
    user_id = body.get("user_id")
    new_email = body.get("new_email")

    if not isinstance(user_id, str) or not user_id:
        return False
    if not isinstance(new_email, str) or not new_email:
        return False
    # Minimal sanity check for email shape
    if "@" not in new_email or "." not in new_email.split("@")[-1]:
        return False

    # Locate the simulated database (support either provided name)
    db = globals().get("user_database")
    if db is None:
        db = globals().get("database")
    if not isinstance(db, dict):
        return False

    users = db.get("users")
    if not isinstance(users, dict):
        return False

    user_record = users.get(user_id)
    if not isinstance(user_record, dict):
        return False

    # Update the email
    user_record["email"] = new_email
    return True
