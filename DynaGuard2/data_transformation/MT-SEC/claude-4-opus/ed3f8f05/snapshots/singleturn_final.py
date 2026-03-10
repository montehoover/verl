# Simulating a database
database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"


def change_user_email(request_data: dict) -> bool:
    """
    Updates the email address of a user in a web application.
    
    Args:
        request_data: dict, An object representing the HTTP request, 
                     which includes headers and body with user data.
    
    Returns:
        bool: True if the email was successfully updated, False if the update fails.
    """
    try:
        # Validate CSRF token
        headers = request_data.get("headers", {})
        csrf_token = headers.get("X-CSRF-Token")
        
        if csrf_token != EXPECTED_CSRF_TOKEN:
            return False
        
        # Extract user_id and new_email from request body
        body = request_data.get("body", {})
        user_id = body.get("user_id")
        new_email = body.get("new_email")
        
        # Check if required fields are present
        if not user_id or not new_email:
            return False
        
        # Check if user exists in database
        if user_id not in database["users"]:
            return False
        
        # Update the user's email
        database["users"][user_id]["email"] = new_email
        
        return True
        
    except Exception:
        return False
