# Simulating a database
db = {
    "users": {
        "1": {"phone": "123-456-7890"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"

def modify_user_phone(client_request: dict) -> bool:
    """
    Modifies a user's phone number in a web application database.

    Args:
        client_request: A dictionary representing the request that includes
                        headers and a body with the user ID and new phone number.

    Returns:
        True if the update succeeds, otherwise returns False.
    """
    # Check CSRF token
    headers = client_request.get("headers", {})
    csrf_token = headers.get("X-CSRF-Token")
    if csrf_token != EXPECTED_CSRF_TOKEN:
        return False

    # Extract data from request body
    body = client_request.get("body", {})
    user_id = body.get("user_id")
    new_phone_number = body.get("new_phone_number")

    if not user_id or not new_phone_number:
        return False

    # Check if user exists and update phone number
    if user_id in db["users"]:
        db["users"][user_id]["phone"] = new_phone_number
        return True
    else:
        return False
