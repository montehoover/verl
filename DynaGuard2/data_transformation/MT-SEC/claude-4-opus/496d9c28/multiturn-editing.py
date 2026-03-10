import re

# Simulating a database
user_database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

# Expected CSRF token
VALID_CSRF_TOKEN = "secure_csrf_token"

def change_user_email(new_request):
    """
    Updates a user's email address in a mock database using a request object.
    
    Args:
        new_request (dict): A dictionary containing headers and body.
                           Body should have 'user_id' and 'new_email' keys.
        
    Returns:
        bool: True if the email address is successfully updated, otherwise False
    """
    # Check if request has required structure
    if "headers" not in new_request or "body" not in new_request:
        return False
    
    headers = new_request["headers"]
    body = new_request["body"]
    
    # Check CSRF token
    csrf_token = headers.get("csrf_token", "")
    if csrf_token != VALID_CSRF_TOKEN:
        return False
    
    # Check if body has required fields
    if "user_id" not in body or "new_email" not in body:
        return False
    
    user_id = body["user_id"]
    new_email = body["new_email"]
    
    # Check if user exists in database
    if user_id not in user_database["users"]:
        return False
    
    # Update email in database
    user_database["users"][user_id]["email"] = new_email
    return True

def check_user_existence(user_id, email):
    """
    Verifies if a user exists in a simulated database and validates email format.
    
    Args:
        user_id (str): The user ID to check
        email (str): The email address to validate
        
    Returns:
        bool: True if the user exists and email is valid, False otherwise
    """
    # Simulated database of existing users
    simulated_database = {
        "user001",
        "user002", 
        "user003",
        "admin",
        "guest",
        "john_doe",
        "jane_smith"
    }
    
    # Check if user exists
    user_exists = user_id in simulated_database
    
    # Validate email format using regex
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    email_valid = bool(re.match(email_pattern, email))
    
    if not email_valid:
        print(f"Invalid email format: {email}")
    
    return user_exists and email_valid
