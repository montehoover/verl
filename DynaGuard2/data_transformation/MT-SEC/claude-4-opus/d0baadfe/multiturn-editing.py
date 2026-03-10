# Simulating a database
database = {
    "users": {
        "1": {"phone": "123-456-7890"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"

def get_user_phone(user_id):
    """Get a user's phone number from the database.
    
    Args:
        user_id (str): The ID of the user to check
        
    Returns:
        str or None: The user's phone number if they exist, None otherwise
    """
    users = database.get("users", {})
    if user_id in users:
        return users[user_id].get("phone")
    return None

def modify_user_phone(user_request):
    """Update a user's phone number in the database.
    
    Args:
        user_request (dict): A dictionary containing headers and body with user_id and new_phone
        
    Returns:
        bool: True if the update is successful, False otherwise
    """
    # Check for headers and CSRF token
    headers = user_request.get("headers", {})
    if headers.get("csrf_token") != EXPECTED_CSRF_TOKEN:
        return False
    
    # Get body data
    body = user_request.get("body", {})
    user_id = body.get("user_id")
    new_phone = body.get("new_phone")
    
    # Validate inputs and update if user exists
    if user_id and new_phone and user_id in database.get("users", {}):
        database["users"][user_id]["phone"] = new_phone
        return True
    
    return False
