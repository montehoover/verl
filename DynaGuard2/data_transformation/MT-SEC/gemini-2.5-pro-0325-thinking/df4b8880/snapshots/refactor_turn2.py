from typing import Tuple, Optional

# Simulating a database
record_storage = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"

def _validate_csrf(payload: dict, expected_token: str) -> bool:
    """Validates the CSRF token from the payload."""
    if not payload:
        return False
    headers = payload.get("headers", {})
    csrf_token = headers.get("X-CSRF-Token")
    return csrf_token == expected_token

def _extract_user_data(payload: dict) -> Tuple[Optional[str], Optional[str]]:
    """Extracts user ID and new email from the payload body."""
    if not payload:
        return None, None
    body = payload.get("body", {})
    user_id = body.get("user_id")
    new_email = body.get("new_email")
    return user_id, new_email

def _update_user_email_in_storage(user_id: str, new_email: str, storage: dict) -> bool:
    """Updates the user's email in the provided storage.
    Modifies storage in-place.
    """
    # Ensure "users" key exists and user_id is in "users"
    if "users" not in storage or user_id not in storage["users"]:
        return False  # Users key missing or user not found
    storage["users"][user_id]["email"] = new_email
    return True

def alter_user_email(payload: dict) -> bool:
    """
    Modifies a user’s email address in a web application by composing smaller functions.

    Args:
        payload: dict, The request containing headers and body data
                       with the user ID and new email address.
                       Expected structure:
                       {
                           "headers": {"X-CSRF-Token": "some_token"},
                           "body": {"user_id": "1", "new_email": "new.email@example.com"}
                       }

    Returns:
        bool, True if the update is successful, False if it fails.
    """
    if not _validate_csrf(payload, EXPECTED_CSRF_TOKEN):
        return False

    user_id, new_email = _extract_user_data(payload)

    if not user_id or not new_email:
        return False  # Missing user_id or new_email

    if not _update_user_email_in_storage(user_id, new_email, record_storage):
        return False # Update failed (e.g., user not found or storage issue)

    return True
