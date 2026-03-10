# Simulating a database
database = {
    "users": {
        "1": {"phone": "123-456-7890"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"


def change_user_phone(request_details: dict) -> bool:
    """
    Modifies a user's phone number in the web application database.
    
    Args:
        request_details: A dictionary representing the request that includes 
                        headers and a body with the user ID and new phone number.
    
    Returns:
        Returns True if the update succeeds, otherwise returns False.
    """
    try:
        # Check if request has required structure
        if not isinstance(request_details, dict):
            return False
        
        headers = request_details.get("headers", {})
        body = request_details.get("body", {})
        
        # Verify CSRF token
        csrf_token = headers.get("X-CSRF-Token")
        if csrf_token != EXPECTED_CSRF_TOKEN:
            return False
        
        # Extract user ID and new phone number
        user_id = body.get("user_id")
        new_phone = body.get("new_phone")
        
        # Validate inputs
        if not user_id or not new_phone:
            return False
        
        # Check if user exists in database
        if user_id not in database.get("users", {}):
            return False
        
        # Update the phone number
        database["users"][user_id]["phone"] = new_phone
        
        return True
        
    except Exception:
        # Handle any unexpected errors
        return False
