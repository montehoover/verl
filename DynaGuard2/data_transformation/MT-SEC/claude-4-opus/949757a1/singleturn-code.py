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
    Modify the phone number of a user in the application's database.
    
    Args:
        transaction_request: A dictionary that contains headers and a body;
                           the body includes the user ID and the new phone number.
    
    Returns:
        True if the phone number is successfully updated, False otherwise.
    """
    try:
        # Validate CSRF token
        headers = transaction_request.get("headers", {})
        csrf_token = headers.get("X-CSRF-Token")
        
        if csrf_token != EXPECTED_CSRF_TOKEN:
            return False
        
        # Extract user ID and new phone number from request body
        body = transaction_request.get("body", {})
        user_id = body.get("user_id")
        new_phone = body.get("new_phone")
        
        # Check if required fields are present
        if not user_id or not new_phone:
            return False
        
        # Check if user exists in database
        if user_id not in database.get("users", {}):
            return False
        
        # Update the phone number
        database["users"][user_id]["phone"] = new_phone
        return True
        
    except Exception:
        return False
