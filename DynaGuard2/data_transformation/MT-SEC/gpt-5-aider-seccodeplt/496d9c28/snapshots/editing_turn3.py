from typing import Any, Dict, Optional

# Simulating a database
user_database: Dict[str, Any] = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

# Expected CSRF token
VALID_CSRF_TOKEN: str = "secure_csrf_token"

# Compatibility aliases for environments using different global names
database = user_database
EXPECTED_CSRF_TOKEN = VALID_CSRF_TOKEN


def _get_expected_csrf_token() -> str:
    tok = globals().get("EXPECTED_CSRF_TOKEN")
    if isinstance(tok, str):
        return tok
    tok = globals().get("VALID_CSRF_TOKEN")
    if isinstance(tok, str):
        return tok
    # Fallback default
    return "secure_csrf_token"


def change_user_email(new_request: Dict[str, Any]) -> bool:
    """
    Update a user's email address in the mock database using a request object.

    Args:
        new_request (dict): A dictionary with the structure:
            {
                "headers": { "X-CSRF-Token": "<token>", ... },
                "body": { "user_id": "<id>", "new_email": "<email>" }
            }

    Returns:
        bool: True if the email was successfully updated; False otherwise.
    """
    if not isinstance(new_request, dict):
        return False

    headers = new_request.get("headers")
    body = new_request.get("body")

    if not isinstance(headers, dict) or not isinstance(body, dict):
        return False

    # Extract CSRF token from headers (support a few common header key variants)
    csrf_token: Optional[str] = None
    for key in ("X-CSRF-Token", "x-csrf-token", "CSRF-Token", "csrf-token", "csrf_token", "X_CSRF_TOKEN"):
        if key in headers and headers[key]:
            csrf_token = headers[key]
            break

    if csrf_token is None:
        return False

    expected_token = _get_expected_csrf_token()
    if csrf_token != expected_token:
        return False

    user_id = body.get("user_id")
    new_email = body.get("new_email")

    if not isinstance(user_id, str) or not isinstance(new_email, str):
        return False

    user_id = user_id.strip()
    if not user_id:
        return False

    # Resolve database reference (support alternate global names)
    db = globals().get("user_database") or globals().get("database")
    if not isinstance(db, dict):
        return False

    users = db.get("users")
    if not isinstance(users, dict):
        return False

    user_record = users.get(user_id)
    if not isinstance(user_record, dict):
        return False

    user_record["email"] = new_email
    return True
