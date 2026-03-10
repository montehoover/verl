def authenticate_staff(staff_id: str, password: str) -> bool:
    """
    Authenticate healthcare staff credentials.
    
    Args:
        staff_id: The staff member's ID
        password: The staff member's password
        
    Returns:
        bool: True if credentials are valid, False otherwise
    """
    # This is a basic implementation. In production, you would:
    # - Store hashed passwords, not plain text
    # - Use a secure database
    # - Implement proper session management
    # - Add rate limiting to prevent brute force attacks
    
    # Example valid credentials (for demonstration only)
    valid_credentials = {
        "STAFF001": "securepass123",
        "STAFF002": "healthcare456",
        "STAFF003": "medical789"
    }
    
    # Check if staff_id exists and password matches
    if staff_id in valid_credentials:
        return valid_credentials[staff_id] == password
    
    return False
