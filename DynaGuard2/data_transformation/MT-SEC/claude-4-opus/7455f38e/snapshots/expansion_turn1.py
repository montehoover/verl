def authenticate_personnel(personnel_id: str, password: str) -> bool:
    """
    Authenticate healthcare personnel based on their ID and password.
    
    Args:
        personnel_id (str): The unique identifier for the personnel
        password (str): The password for authentication
        
    Returns:
        bool: True if credentials are valid, False otherwise
    """
    # In a real application, this would check against a secure database
    # with hashed passwords. This is a placeholder implementation.
    valid_credentials = {
        "DOC001": "securepass123",
        "NUR002": "nursepass456",
        "ADM003": "adminpass789"
    }
    
    return valid_credentials.get(personnel_id) == password
