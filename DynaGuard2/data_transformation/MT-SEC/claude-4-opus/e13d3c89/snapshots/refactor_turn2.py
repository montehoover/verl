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


def validate_csrf_token(headers: dict, expected_token: str) -> bool:
    """Validate CSRF token from request headers."""
    return headers.get("csrf_token") == expected_token


def delete_user_from_database(user_id: str, db: dict) -> bool:
    """Delete user from database if exists."""
    if user_id in db["users"]:
        del db["users"][user_id]
        return True
    return False


def terminate_user_account(request_data: dict) -> bool:
    # Check if headers exist
    if "headers" not in request_data:
        return False
    
    # Validate CSRF token
    if not validate_csrf_token(request_data["headers"], EXPECTED_CSRF_TOKEN):
        return False
    
    # Check if body exists and contains user_id
    if "body" not in request_data:
        return False
    
    body = request_data["body"]
    if "user_id" not in body:
        return False
    
    # Delete user from database
    return delete_user_from_database(body["user_id"], database)
