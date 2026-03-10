def authenticate_staff(staff_id: str, password: str) -> bool:
    """
    Authenticate staff members by checking their credentials.
    
    Args:
        staff_id (str): The staff member's ID
        password (str): The staff member's password
        
    Returns:
        bool: True if credentials are valid, False otherwise
    """
    # Mock staff credentials for demonstration
    # In production, this should check against a secure database
    # with hashed passwords
    valid_credentials = {
        "STAFF001": "securepass123",
        "STAFF002": "medical456",
        "STAFF003": "health789"
    }
    
    # Check if staff_id exists and password matches
    if staff_id in valid_credentials:
        return valid_credentials[staff_id] == password
    
    return False
