# Simulating a database
user_data = {
    "users": {
        "1": {"mobile": "123-456-7891"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "known_csrf_token"

def alter_user_phone(input: dict) -> bool:
    """
    Updates the phone number of a specific user.

    Args:
        input: A request dictionary containing headers with the CSRF token
               and a body with the user ID and the new phone number.

    Returns:
        True if the phone number is successfully updated, False otherwise.
    """
    # Validate CSRF token
    headers = input.get("headers", {})
    csrf_token = headers.get("X-CSRF-Token")
    if csrf_token != EXPECTED_CSRF_TOKEN:
        return False

    # Get user ID and new phone number from request body
    body = input.get("body", {})
    user_id = body.get("user_id")
    new_phone_number = body.get("new_phone_number")

    # Guard clause for missing user_id or new_phone_number
    if not user_id or not new_phone_number:
        return False

    # Guard clause for user not found
    if user_id not in user_data["users"]:
        return False

    # Update phone number in the database
    user_data["users"][user_id]["mobile"] = new_phone_number
    return True
