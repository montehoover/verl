# Simulating a database
record_storage = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"


def alter_user_email(payload: dict) -> bool:
    """
    Modify a user's email address in the web application.
    
    Args:
        payload: dict containing headers and body data with user ID and new email
        
    Returns:
        bool: True if update is successful, False if it fails
    """
    try:
        # Validate CSRF token
        headers = payload.get("headers", {})
        csrf_token = headers.get("X-CSRF-Token")
        
        if csrf_token != EXPECTED_CSRF_TOKEN:
            return False
            
        # Extract user_id and new_email from body
        body = payload.get("body", {})
        user_id = body.get("user_id")
        new_email = body.get("new_email")
        
        # Validate required fields
        if not user_id or not new_email:
            return False
            
        # Check if user exists in database
        if user_id not in record_storage["users"]:
            return False
            
        # Update user's email
        record_storage["users"][user_id]["email"] = new_email
        return True
        
    except Exception:
        return False
