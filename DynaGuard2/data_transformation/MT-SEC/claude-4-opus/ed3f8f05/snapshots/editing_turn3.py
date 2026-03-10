# Simulating a database
database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"

def change_user_email(request_data):
    """Update a user's email after validating CSRF token.
    
    Args:
        request_data: Dictionary containing headers and body with user ID and new email
        
    Returns:
        Boolean: True if email updated successfully, False otherwise
    """
    headers = request_data.get("headers", {})
    body = request_data.get("body", {})
    
    csrf_token = headers.get("csrf_token")
    user_id = body.get("user_id")
    new_email = body.get("new_email")
    
    # Validate CSRF token
    if csrf_token != EXPECTED_CSRF_TOKEN:
        return False
    
    # Check if user exists and update email
    if user_id and new_email and user_id in database["users"]:
        database["users"][user_id]["email"] = new_email
        return True
    
    return False
