# Simulating a database
database = {
    "users": {
        "1": {"phone": "123-456-7890"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"

def change_user_contact(transaction_request: dict) -> bool:
    """
    Modifies the phone number of a user in the application's database.

    Args:
        transaction_request: A dictionary that contains headers and a body;
                             the body includes the user ID and the new phone number.

    Returns:
        True if the phone number is successfully updated, False otherwise.
    """
    try:
        body = transaction_request.get("body", {})
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
        # Log the exception in a real application
        return False
