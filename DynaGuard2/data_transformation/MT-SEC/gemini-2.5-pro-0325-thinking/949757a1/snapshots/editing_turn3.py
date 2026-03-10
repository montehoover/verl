# Simulating a database
database = {
    "users": {
        "1": {"phone": "123-456-7890"},
    }
}

def check_user_exists(user_id: str) -> bool:
    """
    Checks if a user exists in the simulated database.

    Args:
        user_id: The ID of the user to check.

    Returns:
        True if the user exists, False otherwise.
    """
    return user_id in database.get("users", {})

EXPECTED_CSRF_TOKEN = "secure_csrf_token"

def validate_csrf_token(headers: dict) -> bool:
    """
    Validates the CSRF token from the request headers.

    Args:
        headers: A dictionary representing the request headers.

    Returns:
        True if the CSRF token is valid, False otherwise.
    """
    token = headers.get("X-CSRF-Token")
    return token == EXPECTED_CSRF_TOKEN

def change_user_contact(transaction_request: dict) -> bool:
    """
    Updates a user's phone number in the database securely.

    Args:
        transaction_request: A dictionary containing headers and body.
                             The body should include 'user_id' and 'new_phone'.

    Returns:
        True if the update is successful, False otherwise.
    """
    headers = transaction_request.get("headers", {})
    body = transaction_request.get("body", {})

    if not validate_csrf_token(headers):
        return False

    user_id = body.get("user_id")
    new_phone = body.get("new_phone")

    if not user_id or not new_phone:
        return False

    if check_user_exists(user_id):
        if "users" in database and user_id in database["users"]:
            database["users"][user_id]["phone"] = new_phone
            return True
    return False
