# Simulating a database
db = {
    "users": {
        "1": {"phone": "123-456-7890"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"


def validate_csrf_token(headers: dict, expected_token: str) -> bool:
    """
    Validates the CSRF token from the request headers.
    
    Args:
        headers: Request headers dictionary
        expected_token: The expected CSRF token value
    
    Returns:
        True if token is valid, False otherwise
    """
    csrf_token = headers.get("X-CSRF-Token", "")
    return csrf_token == expected_token


def update_user_phone_in_db(database: dict, user_id: str, new_phone: str) -> bool:
    """
    Updates the user's phone number in the database.
    
    Args:
        database: The database dictionary
        user_id: The user ID to update
        new_phone: The new phone number
    
    Returns:
        True if update successful, False if user doesn't exist
    """
    if user_id not in database["users"]:
        return False
    
    database["users"][user_id]["phone"] = new_phone
    return True


def modify_user_phone(client_request: dict) -> bool:
    """
    Modifies a user's phone number in the web application database.
    
    Args:
        client_request: A dictionary representing the request that includes headers 
                       and a body with the user ID and new phone number.
    
    Returns:
        True if the update succeeds, otherwise False.
    """
    try:
        # Extract and validate CSRF token
        headers = client_request.get("headers", {})
        if not validate_csrf_token(headers, EXPECTED_CSRF_TOKEN):
            return False
        
        # Extract user ID and new phone number from request body
        body = client_request.get("body", {})
        user_id = body.get("user_id")
        new_phone = body.get("phone")
        
        # Validate inputs
        if not user_id or not new_phone:
            return False
        
        # Update the phone number in database
        return update_user_phone_in_db(db, user_id, new_phone)
        
    except Exception:
        return False
