def check_user_existence(user_id):
    """
    Verifies if a user exists in a simulated database.
    
    Args:
        user_id (str): The user ID to check
        
    Returns:
        bool: True if the user exists, False otherwise
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
    
    return user_id in simulated_database
