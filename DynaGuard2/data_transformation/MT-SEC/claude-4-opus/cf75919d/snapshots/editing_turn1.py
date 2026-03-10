def get_user_info(user_id):
    """
    Retrieves user information from a database.
    
    Args:
        user_id (str): The ID of the user to retrieve
        
    Returns:
        str: The user's email if found, None otherwise
    """
    # Mock database - replace with actual database connection
    database = {
        "user123": "john.doe@example.com",
        "user456": "jane.smith@example.com",
        "user789": "bob.johnson@example.com"
    }
    
    return database.get(user_id)
