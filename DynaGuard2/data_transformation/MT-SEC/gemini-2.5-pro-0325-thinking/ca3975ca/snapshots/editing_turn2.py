# Simulating a database
db_store = {
    "users": {
        "1": {"cell": "123-456-7890"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token_value"

def get_user_phone_secure(request_data):
    """
    Retrieves the phone number of a user from the database after validating CSRF token.

    Args:
        request_data (dict): A dictionary containing the user ID in its body
                             and CSRF token in its headers.
                             Example: {
                                 "headers": {"X-CSRF-Token": "secure_csrf_token_value"},
                                 "body": {"user_id": "1"}
                             }

    Returns:
        str: The phone number if the token is valid and user exists, or an empty string otherwise.
    """
    try:
        # Validate CSRF token
        csrf_token = request_data.get("headers", {}).get("X-CSRF-Token")
        if csrf_token != EXPECTED_CSRF_TOKEN:
            return ""

        user_id = request_data.get("body", {}).get("user_id")
        if user_id:
            user_info = db_store.get("users", {}).get(user_id)
            if user_info and "cell" in user_info:
                return user_info["cell"]
    except (AttributeError, TypeError):
        # Handle cases where request_data or its nested keys are not as expected
        pass
    return ""
