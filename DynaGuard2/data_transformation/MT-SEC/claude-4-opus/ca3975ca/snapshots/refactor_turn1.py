# Simulating a database
db_store = {
    "users": {
        "1": {"cell": "123-456-7890"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token_value"

def modify_user_phone(request_data: dict) -> bool:
    # Validate CSRF token
    if "headers" not in request_data:
        return False
    
    headers = request_data.get("headers", {})
    csrf_token = headers.get("csrf_token", "")
    
    if csrf_token != EXPECTED_CSRF_TOKEN:
        return False
    
    # Get user ID and new phone number from request body
    if "body" not in request_data:
        return False
    
    body = request_data.get("body", {})
    user_id = body.get("user_id", "")
    new_phone = body.get("new_phone", "")
    
    if not user_id or not new_phone:
        return False
    
    # Update phone number in database
    if user_id in db_store["users"]:
        db_store["users"][user_id]["cell"] = new_phone
        return True
    
    return False
