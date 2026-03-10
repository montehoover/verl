# Simulating a database
database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"

def change_user_email(request_data: dict) -> bool:
    # Validate CSRF token
    headers = request_data.get("headers", {})
    csrf_token = headers.get("X-CSRF-Token")
    
    if csrf_token != EXPECTED_CSRF_TOKEN:
        return False
    
    # Get user data from request body
    body = request_data.get("body", {})
    user_id = body.get("user_id")
    new_email = body.get("email")
    
    # Validate required fields
    if not user_id or not new_email:
        return False
    
    # Check if user exists in database
    if user_id not in database["users"]:
        return False
    
    # Update user email
    database["users"][user_id]["email"] = new_email
    return True
