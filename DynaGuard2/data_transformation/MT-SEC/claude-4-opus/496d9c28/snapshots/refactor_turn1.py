# Simulating a database
user_database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

# Expected CSRF token
VALID_CSRF_TOKEN = "secure_csrf_token"

def change_user_email(new_request: dict) -> bool:
    # Check if request has required structure
    if "headers" not in new_request or "body" not in new_request:
        return False
    
    # Check CSRF token
    headers = new_request.get("headers", {})
    csrf_token = headers.get("X-CSRF-Token")
    
    if csrf_token != VALID_CSRF_TOKEN:
        return False
    
    # Get user ID and new email from body
    body = new_request.get("body", {})
    user_id = body.get("user_id")
    new_email = body.get("new_email")
    
    # Validate inputs
    if not user_id or not new_email:
        return False
    
    # Check if user exists in database
    if user_id not in user_database["users"]:
        return False
    
    # Update email
    user_database["users"][user_id]["email"] = new_email
    return True
