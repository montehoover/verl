# Simulating a database
database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

def check_user_email(request_data):
    """Check if a given email matches the user's current email in the database.
    
    Args:
        request_data: Dictionary containing a body with the user ID and email to check
        
    Returns:
        Boolean: True if email matches the current email, False otherwise
    """
    body = request_data.get("body", {})
    user_id = body.get("user_id")
    email_to_check = body.get("email")
    
    if user_id and email_to_check and user_id in database["users"]:
        return database["users"][user_id]["email"] == email_to_check
    
    return False
