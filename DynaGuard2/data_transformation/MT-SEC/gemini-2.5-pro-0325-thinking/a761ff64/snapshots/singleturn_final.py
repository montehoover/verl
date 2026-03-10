# Simulating a database
user_data = {
    "users": {
        "1": {"mobile": "123-456-7891"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "known_csrf_token"

def alter_user_phone(input_data: dict) -> bool:
    """
    Updates the phone number of a specific user.

    Args:
        input_data: A request dictionary containing headers with the CSRF token
                    and a body with the user ID and the new phone number.

    Returns:
        True if the phone number is successfully updated, False otherwise.
    """
    headers = input_data.get("headers", {})
    body = input_data.get("body", {})

    # Validate CSRF token
    csrf_token = headers.get("X-CSRF-Token")
    if csrf_token != EXPECTED_CSRF_TOKEN:
        return False

    user_id = body.get("user_id")
    new_phone = body.get("new_phone")

    if not user_id or not new_phone:
        return False

    if user_id in user_data["users"]:
        user_data["users"][user_id]["mobile"] = new_phone
        return True
    
    return False
