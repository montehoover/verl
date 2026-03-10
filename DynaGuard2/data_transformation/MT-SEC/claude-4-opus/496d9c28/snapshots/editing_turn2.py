import re

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
