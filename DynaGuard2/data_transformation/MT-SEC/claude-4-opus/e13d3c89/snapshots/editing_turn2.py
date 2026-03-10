def get_user_info(user_id, csrf_token):
    # Mock database - in a real application, this would query an actual database
    database = {
        "user123": "john.doe@example.com",
        "user456": "jane.smith@example.com",
        "user789": "bob.johnson@example.com"
    }
    
    # Mock CSRF tokens - in a real application, these would be dynamically generated and validated
    valid_csrf_tokens = {
        "token_abc123",
        "token_def456",
        "token_ghi789"
    }
    
    # Check if CSRF token is valid
    if csrf_token not in valid_csrf_tokens:
        return None
    
    return database.get(user_id)
