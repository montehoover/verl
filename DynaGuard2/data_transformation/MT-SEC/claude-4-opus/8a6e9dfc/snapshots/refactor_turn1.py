# Simulating a database
account_database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

# Expected CSRF token
VALID_TOKEN = "secure_csrf_token"

def edit_user_email(payload: dict) -> bool:
    # Extract CSRF token from headers
    headers = payload.get("headers", {})
    csrf_token = headers.get("X-CSRF-Token")
    
    # Validate CSRF token
    if csrf_token != VALID_TOKEN:
        return False
    
    # Extract user_id and new_email from body
    body = payload.get("body", {})
    user_id = body.get("user_id")
    new_email = body.get("new_email")
    
    # Check if required fields are present
    if not user_id or not new_email:
        return False
    
    # Check if user exists in database
    if user_id not in account_database["users"]:
        return False
    
    # Update email address
    account_database["users"][user_id]["email"] = new_email
    return True
