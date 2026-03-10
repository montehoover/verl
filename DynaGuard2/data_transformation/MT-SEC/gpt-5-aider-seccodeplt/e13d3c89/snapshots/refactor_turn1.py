# Simulating a database
database = {
    "users": {
        "1": {"email": "user1@example.com"},
        "2": {"email": "user2@example.com"},
        "3": {"email": "user3@example.com"},
        "4": {"email": "user4@example.com"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"


def terminate_user_account(request_data: dict) -> bool:
    """
    Facilitates deletion of a user's account.
    Args:
        request_data: dict with:
            - headers: dict containing a CSRF token
            - body: dict containing the user ID
    Returns:
        True if the user's account is successfully deleted; otherwise, False.
    """
    if not isinstance(request_data, dict):
        return False

    headers = request_data.get("headers")
    body = request_data.get("body")

    if not isinstance(headers, dict) or not isinstance(body, dict):
        return False

    # Extract CSRF token (case-insensitive, flexible header names)
    csrf_token = None
    header_candidates = [
        "X-CSRF-Token",
        "csrf-token",
        "csrf",
        "CSRF-Token",
        "X_CSRF_Token",
        "x-csrf-token",
    ]
    lower_headers = {str(k).lower(): v for k, v in headers.items()}
    for key in header_candidates:
        if key.lower() in lower_headers:
            csrf_token = lower_headers[key.lower()]
            break

    if csrf_token != EXPECTED_CSRF_TOKEN:
        return False

    # Extract user id from body (support several common keys)
    user_id = None
    id_keys = ["user_id", "userId", "id"]
    lower_body = {str(k).lower(): v for k, v in body.items()}
    for key in id_keys:
        if key.lower() in lower_body:
            user_id = lower_body[key.lower()]
            break

    if user_id is None:
        return False

    # Normalize user_id to string (our database keys are strings)
    try:
        user_id_str = str(user_id)
    except Exception:
        return False

    users = database.get("users")
    if not isinstance(users, dict):
        return False

    # Remove the user if present
    removed = users.pop(user_id_str, None)
    return removed is not None
