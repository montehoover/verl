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


def validate_csrf_token(headers: dict) -> bool:
    """Validate the CSRF token from headers."""
    csrf_token = headers.get("csrf_token", "")
    return csrf_token == EXPECTED_CSRF_TOKEN


def extract_user_id(body: dict) -> str:
    """Extract user ID from request body."""
    return body.get("user_id", "")


def delete_user_from_database(user_id: str) -> bool:
    """Delete user from database if exists."""
    if user_id in database["users"]:
        del database["users"][user_id]
        return True
    return False


def remove_user_account(input_request: dict) -> bool:
    # Extract headers and body
    headers = input_request.get("headers", {})
    body = input_request.get("body", {})
    
    # Pipeline: validate -> extract -> delete
    if not validate_csrf_token(headers):
        return False
    
    user_id = extract_user_id(body)
    if not user_id:
        return False
    
    return delete_user_from_database(user_id)
