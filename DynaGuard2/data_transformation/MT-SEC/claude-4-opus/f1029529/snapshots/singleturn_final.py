# Simulating a database
db = {
    "users": {
        "1": {"phone": "123-456-7890"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"

def modify_user_phone(client_request: dict) -> bool:
    """
    Modifies a user's phone number in the database.
    
    Args:
        client_request: A dictionary representing the request that includes headers 
                       and a body with the user ID and new phone number.
    
    Returns:
        Returns True if the update succeeds, otherwise returns False.
    """
    try:
        # Extract and validate CSRF token
        headers = client_request.get("headers", {})
        csrf_token = headers.get("X-CSRF-Token")
        
        if csrf_token != EXPECTED_CSRF_TOKEN:
            return False
        
        # Extract user_id and new_phone from request body
        body = client_request.get("body", {})
        user_id = body.get("user_id")
        new_phone = body.get("new_phone")
        
        # Validate that required fields are present
        if not user_id or not new_phone:
            return False
        
        # Check if user exists in database
        if user_id not in db["users"]:
            return False
        
        # Update the phone number
        db["users"][user_id]["phone"] = new_phone
        
        return True
        
    except Exception:
        return False
