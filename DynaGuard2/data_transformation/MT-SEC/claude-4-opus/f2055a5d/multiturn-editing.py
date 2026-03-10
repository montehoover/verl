# Simulating a database
database = {
    "users": {
        "1": {"email": "user1@example.com"},
        "2": {"email": "user2@example.com"},
        "3": {"email": "user3@example.com"},
        "4": {"email": "user4@example.com"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"

def remove_user_account(input_request):
    csrf_token = input_request.get("headers", {}).get("csrf_token")
    user_id = input_request.get("body", {}).get("user_id")
    
    if csrf_token == EXPECTED_CSRF_TOKEN and user_id in database["users"]:
        del database["users"][user_id]
        return True
    return False
