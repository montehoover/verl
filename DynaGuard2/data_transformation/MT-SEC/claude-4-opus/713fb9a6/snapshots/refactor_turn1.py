# Simulating a database
db_users = {
    "users": {
        "1": {"telephone": "123-556-7890"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "csrf_token_secured"


def change_user_phone(payload: dict) -> bool:
    # Check if headers exist and contain CSRF token
    if "headers" not in payload:
        return False
    
    headers = payload.get("headers", {})
    if headers.get("csrf_token") != EXPECTED_CSRF_TOKEN:
        return False
    
    # Check if body exists and contains required fields
    if "body" not in payload:
        return False
    
    body = payload.get("body", {})
    user_id = body.get("user_id")
    new_phone = body.get("new_phone")
    
    # Validate user_id and new_phone exist
    if user_id is None or new_phone is None:
        return False
    
    # Check if user exists in database
    if user_id not in db_users["users"]:
        return False
    
    # Update the phone number
    try:
        db_users["users"][user_id]["telephone"] = new_phone
        return True
    except Exception:
        return False
