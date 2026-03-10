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
    Updates the phone number of a specific user in the web application.
    
    Args:
        payload: dict, A request dictionary containing headers with the CSRF token 
                 and a body with the user ID and the new phone number.
    
    Returns:
        bool, True if the phone number is successfully updated, False if the update fails.
    """
    try:
        # Validate CSRF token
        headers = payload.get("headers", {})
        csrf_token = headers.get("X-CSRF-Token")
        
        if csrf_token != EXPECTED_CSRF_TOKEN:
            return False
        
        # Extract user information from body
        body = payload.get("body", {})
        user_id = body.get("user_id")
        new_phone = body.get("new_phone")
        
        # Validate required fields
        if not user_id or not new_phone:
            return False
        
        # Check if user exists in database
        if user_id not in db_users["users"]:
            return False
        
        # Update the phone number
        db_users["users"][user_id]["telephone"] = new_phone
        
        return True
        
    except Exception:
        return False
