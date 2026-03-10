def get_user_info(user_id, auth_token):
    # Mock authentication token - replace with actual authentication system
    VALID_TOKEN = "secret-auth-token-12345"
    
    # Check if auth token is valid
    if auth_token != VALID_TOKEN:
        return None
    
    # Mock database - replace with actual database connection
    database = {
        "user123": {"email": "user123@example.com"},
        "user456": {"email": "user456@example.com"},
        "user789": {"email": "user789@example.com"}
    }
    
    if user_id in database:
        return {"email": database[user_id]["email"]}
    else:
        return None
