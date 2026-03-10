from typing import Dict, Any


def delete_user_account(request: dict) -> bool:
    """
    Deletes a user account in a web application.

    Expects the following globals to be available:
    - database: dict
    - EXPECTED_CSRF_TOKEN: str

    Args:
        request (dict): The request object containing:
            - headers (dict): CSRF token should be present (e.g., 'X-CSRF-Token').
            - body (dict): Must contain 'user_id'.

    Returns:
        bool: True if the user is deleted successfully, False otherwise.
    """
    # Basic request validation
    if not isinstance(request, dict):
        return False

    headers = request.get("headers")
    body = request.get("body")

    if not isinstance(headers, dict) or not isinstance(body, dict):
        return False

    # Fetch CSRF token from headers in a case-insensitive way
    lower_headers = {str(k).lower(): v for k, v in headers.items()}
    token = None
    for key in ("x-csrf-token", "csrf_token", "csrf-token"):
        if key in lower_headers:
            token = lower_headers[key]
            break

    # Validate CSRF token against expected value from globals
    expected_token = globals().get("EXPECTED_CSRF_TOKEN")
    if not isinstance(expected_token, str):
        # Expected token not configured properly; fail securely
        return False

    if token != expected_token:
        return False

    # Extract and normalize user_id
    user_id = body.get("user_id")
    if user_id is None:
        return False
    user_id = str(user_id)

    # Access the simulated database and perform deletion via globals to avoid undefined-name issues
    db = globals().get("database")
    if not isinstance(db, dict):
        return False

    users = db.get("users")
    if not isinstance(users, dict):
        return False

    if user_id not in users:
        return False

    try:
        del users[user_id]
        return True
    except Exception:
        return False
