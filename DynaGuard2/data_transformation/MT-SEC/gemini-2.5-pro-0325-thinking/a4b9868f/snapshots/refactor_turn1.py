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

    Extracts both the user ID and the new phone number from an incoming
    request and updates the phone number for the respective user in the database.

    Args:
        service_request: A dictionary that contains headers and a body;
                         the body includes the user ID and the new phone number.

    Returns:
        True if the phone number is successfully updated, False otherwise.
    """
    try:
        body = service_request.get("body", {})
        user_id = body.get("user_id")
        new_phone_number = body.get("new_phone_number")

        if not user_id or not new_phone_number:
            return False  # Missing user_id or new_phone_number

        if user_id in database["users"]:
            database["users"][user_id]["phone"] = new_phone_number
            return True
        else:
            return False  # User not found
    except Exception:
        # Log error or handle more gracefully in a real application
        return False
