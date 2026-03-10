# Simulating a database
email_database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"

def retrieve_user_email(user_request):
    user_id = user_request.get("body", {}).get("user_id", "")
    if user_id in email_database["users"]:
        return email_database["users"][user_id]["email"]
    return ""

def check_user_email(user_request):
    body = user_request.get("body", {})
    user_id = body.get("user_id", "")
    email_to_check = body.get("email", "")
    
    if user_id in email_database["users"]:
        return email_database["users"][user_id]["email"] == email_to_check
    return False

def modify_user_email_address(user_request):
    headers = user_request.get("headers", {})
    body = user_request.get("body", {})
    
    csrf_token = headers.get("csrf_token", "")
    user_id = body.get("user_id", "")
    new_email = body.get("new_email", "")
    
    if csrf_token == EXPECTED_CSRF_TOKEN and user_id in email_database["users"]:
        email_database["users"][user_id]["email"] = new_email
        return True
    return False
