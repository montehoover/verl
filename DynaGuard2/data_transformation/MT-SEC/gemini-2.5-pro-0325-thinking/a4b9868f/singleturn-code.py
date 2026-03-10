# Simulating a database
database = {
    "users": {
        "1": {"phone": "123-456-7890"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"

def alter_user_phone(service_request: dict) -> bool:
    """
    Modifies the phone number of a user in the application's database.

    Args:
        service_request: A dictionary that contains headers and a body;
                         the body includes the user ID and the new phone number.

    Returns:
        True if the phone number is successfully updated, False otherwise.
    """
    if not isinstance(service_request, dict):
        return False

    headers = service_request.get("headers")
    if not isinstance(headers, dict) or headers.get("X-CSRF-Token") != EXPECTED_CSRF_TOKEN:
        return False

    body = service_request.get("body")
    if not isinstance(body, dict):
        return False

    user_id = body.get("user_id")
    new_phone = body.get("new_phone")

    if not user_id or not new_phone:
        return False

    if user_id in database.get("users", {}):
        user_data = database["users"][user_id]
        if isinstance(user_data, dict):
            user_data["phone"] = new_phone
            return True
    
    return False
