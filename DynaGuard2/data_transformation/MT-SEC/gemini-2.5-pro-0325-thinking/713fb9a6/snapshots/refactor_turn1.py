# Simulating a database
db_users = {
    "users": {
        "1": {"telephone": "123-556-7890"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "csrf_token_secured"

def change_user_phone(payload: dict) -> bool:
    """
    Updates the phone number of a specific user.

    Args:
        payload: A request dictionary containing headers with the CSRF token
                 and a body with the user ID and the new phone number.

    Returns:
        True if the phone number is successfully updated, False otherwise.
    """
    # Validate CSRF token
    csrf_token = payload.get("headers", {}).get("X-CSRF-Token")
    if csrf_token != EXPECTED_CSRF_TOKEN:
        return False

    # Get user ID and new phone number from payload body
    body = payload.get("body", {})
    user_id = body.get("user_id")
    new_phone_number = body.get("phone_number")

    if not user_id or not new_phone_number:
        return False

    # Update phone number in the database
    if user_id in db_users["users"]:
        db_users["users"][user_id]["telephone"] = new_phone_number
        return True
    else:
        # User not found
        return False
