from typing import Any, Dict

# Simulating a database
db = {
    "users": {
        "1": {"phone": "123-456-7890"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"


def modify_user_phone(client_request: Dict[str, Any]) -> bool:
    """
    Update a user's phone number in the mock database after validating CSRF token.

    Args:
        client_request (dict): A request-like dictionary with:
            - headers (dict): Must contain a valid CSRF token.
            - body (dict): Must contain 'user_id' and 'new_phone'.

    Returns:
        bool: True if the user's phone was updated, False otherwise.
    """
    if not isinstance(client_request, dict):
        return False

    headers = client_request.get("headers") or {}
    if not isinstance(headers, dict):
        return False

    # Accept common CSRF header key variations
    token = None
    for key in ("X-CSRF-Token", "x-csrf-token", "CSRF-Token", "csrf-token", "csrf_token", "csrf"):
        if key in headers:
            token = headers[key]
            break

    if token != EXPECTED_CSRF_TOKEN:
        return False

    body = client_request.get("body") or {}
    if not isinstance(body, dict):
        return False

    if "user_id" not in body or "new_phone" not in body:
        return False

    user_id = str(body["user_id"])
    new_phone = body["new_phone"]

    if not isinstance(new_phone, str):
        return False

    users = db.get("users")
    if not isinstance(users, dict):
        return False

    user_record = users.get(user_id)
    if not isinstance(user_record, dict):
        return False

    user_record["phone"] = new_phone
    return True
