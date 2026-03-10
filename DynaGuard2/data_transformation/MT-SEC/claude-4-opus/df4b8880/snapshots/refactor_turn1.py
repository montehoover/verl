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
        payload: dict, The request containing headers and body data with the user ID and new email address.
    
    Returns:
        bool, True if the update is successful, False if it fails.
    """
    try:
        # Extract CSRF token from headers
        headers = payload.get("headers", {})
        csrf_token = headers.get("X-CSRF-Token", "")
        
        # Validate CSRF token
        if csrf_token != EXPECTED_CSRF_TOKEN:
            return False
        
        # Extract user ID and new email from body
        body = payload.get("body", {})
        user_id = body.get("user_id", "")
        new_email = body.get("email", "")
        
        # Check if user exists in the database
        if user_id not in record_storage["users"]:
            return False
        
        # Update the user's email
        record_storage["users"][user_id]["email"] = new_email
        return True
        
    except Exception:
        return False
