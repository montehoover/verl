def authenticate_worker(worker_id: str, password: str) -> bool:
    """
    Authenticates a healthcare worker based on their ID and password.
    
    Args:
        worker_id (str): The worker's unique identifier
        password (str): The worker's password
        
    Returns:
        bool: True if credentials are valid, False otherwise
    """
    # This is a placeholder implementation. In production, you would:
    # - Hash the password before comparing
    # - Query a secure database
    # - Use proper authentication mechanisms
    
    # Temporary hardcoded credentials for demonstration
    valid_credentials = {
        "HW001": "securepass123",
        "HW002": "healthpass456",
        "HW003": "medicalauth789"
    }
    
    # Check if worker_id exists and password matches
    if worker_id in valid_credentials:
        return valid_credentials[worker_id] == password
    
    return False
