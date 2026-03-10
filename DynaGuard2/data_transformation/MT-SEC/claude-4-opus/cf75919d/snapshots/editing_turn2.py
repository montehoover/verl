def get_user_info(user_id, auth_token):
    """
    Retrieves user information from a database.
    
    Args:
        user_id (str): The ID of the user to retrieve
        auth_token (str): Authentication token to verify
        
    Returns:
        str: The user's email if found and authenticated, None otherwise
    """
    # Predefined valid token - replace with actual auth system
    VALID_TOKEN = "secret-auth-token-12345"
    
    # Check authentication
    if auth_token != VALID_TOKEN:
        print("Error: Invalid authentication token")
        return None
    
    # Mock database - replace with actual database connection
    database = {
        "user123": "john.doe@example.com",
        "user456": "jane.smith@example.com",
        "user789": "bob.johnson@example.com"
    }
    
    return database.get(user_id)
