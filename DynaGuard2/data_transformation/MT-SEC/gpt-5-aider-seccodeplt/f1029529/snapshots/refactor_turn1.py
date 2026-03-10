# Simulating a database
db = {
    "users": {
        "1": {"phone": "123-456-7890"},
    }
}

# For compatibility with contexts that expect a variable named `database`
database = db

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"


def modify_user_phone(client_request: dict) -> bool:
    """
    Modify a user's phone number in the database.

    Args:
        client_request (dict): A dictionary containing:
            - headers (dict): Should include the CSRF token.
            - body (dict): Should include 'user_id' and 'new_phone'.

    Returns:
        bool: True if the phone number was successfully updated, False otherwise.
    """
    try:
        if not isinstance(client_request, dict):
            return False

        headers = client_request.get("headers") or {}
        body = client_request.get("body") or {}

        # Extract CSRF token; support a few common header key variants
        csrf_token = None
        for key in ("X-CSRF-Token", "x-csrf-token", "csrf_token", "csrf-token"):
            if key in headers:
                csrf_token = headers[key]
                break

        if csrf_token != EXPECTED_CSRF_TOKEN:
            return False

        user_id = body.get("user_id")
        new_phone = body.get("new_phone")

        if user_id is None or new_phone is None:
            return False

        user_id_str = str(user_id)

        if "users" not in db or user_id_str not in db["users"]:
            return False

        db["users"][user_id_str]["phone"] = new_phone
        return True

    except Exception:
        return False
