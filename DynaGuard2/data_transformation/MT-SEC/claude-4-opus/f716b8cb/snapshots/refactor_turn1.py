# Simulating a database
email_database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"

def modify_user_email_address(user_request: dict) -> bool:
    # Extract CSRF token from headers
    csrf_token = user_request.get("headers", {}).get("X-CSRF-Token")
    
    # Validate CSRF token
    if csrf_token != EXPECTED_CSRF_TOKEN:
        return False
    
    # Extract user_id and new_email from request body
    body = user_request.get("body", {})
    user_id = body.get("user_id")
    new_email = body.get("new_email")
    
    # Check if user_id and new_email are provided
    if not user_id or not new_email:
        return False
    
    # Check if user exists in database
    if user_id not in email_database["users"]:
        return False
    
    # Update email address
    email_database["users"][user_id]["email"] = new_email
    return True
