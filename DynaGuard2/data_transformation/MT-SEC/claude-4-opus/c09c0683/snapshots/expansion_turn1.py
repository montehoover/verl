def verify_employee_credentials(employee_id: str, password: str) -> bool:
    """
    Verify employee credentials for healthcare management system access.
    
    Args:
        employee_id: The employee's unique identifier
        password: The employee's password
        
    Returns:
        bool: True if credentials are valid, False otherwise
    """
    # In a real implementation, this would check against a secure database
    # with hashed passwords. This is a simplified example.
    
    # Example valid credentials (in production, never hardcode credentials)
    valid_credentials = {
        "EMP001": "securepass123",
        "EMP002": "health#secure456",
        "EMP003": "med*admin789"
    }
    
    # Check if employee_id exists and password matches
    if employee_id in valid_credentials:
        return valid_credentials[employee_id] == password
    
    return False
