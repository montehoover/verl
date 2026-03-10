# Simulating a database
database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

def get_user_email(request_data):
    """Retrieve a user's email from the database.
    
    Args:
        request_data: Dictionary containing a body with the user ID
        
    Returns:
        Email as a string if user is found, empty string otherwise
    """
    user_id = request_data.get("body", {}).get("user_id")
    
    if user_id and user_id in database["users"]:
        return database["users"][user_id]["email"]
    
    return ""
