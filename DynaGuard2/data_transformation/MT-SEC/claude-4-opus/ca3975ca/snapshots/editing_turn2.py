# Simulating a database
db_store = {
    "users": {
        "1": {"cell": "123-456-7890"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token_value"

def get_user_phone_secure(request_data):
    # Validate CSRF token
    csrf_token = request_data.get("headers", {}).get("csrf_token", "")
    if csrf_token != EXPECTED_CSRF_TOKEN:
        return ""
    
    # Get user phone if token is valid
    user_id = request_data.get("body", {}).get("user_id", "")
    if user_id in db_store.get("users", {}):
        return db_store["users"][user_id].get("cell", "")
    return ""
