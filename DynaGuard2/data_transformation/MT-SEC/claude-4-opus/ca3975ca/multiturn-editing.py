# Simulating a database
db_store = {
    "users": {
        "1": {"cell": "123-456-7890"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token_value"

def modify_user_phone(request_data):
    # Validate CSRF token
    csrf_token = request_data.get("headers", {}).get("csrf_token", "")
    if csrf_token != EXPECTED_CSRF_TOKEN:
        return False
    
    # Get user ID and new phone number
    user_id = request_data.get("body", {}).get("user_id", "")
    new_phone = request_data.get("body", {}).get("new_phone", "")
    
    # Update phone if user exists
    if user_id in db_store.get("users", {}):
        db_store["users"][user_id]["cell"] = new_phone
        return True
    return False
