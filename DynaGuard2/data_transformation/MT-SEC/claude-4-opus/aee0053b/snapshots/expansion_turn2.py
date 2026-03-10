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


def check_staff_permission(staff_num: str, patient_ref: str) -> bool:
    """
    Check if a staff member has permission to access a patient's record.
    
    Args:
        staff_num: The staff member's ID number
        patient_ref: The patient's reference number
        
    Returns:
        bool: True if staff member has permission, False otherwise
    """
    # Example permission mapping
    # In production, this would be retrieved from a database
    staff_permissions = {
        "STAFF001": ["PAT001", "PAT002", "PAT003", "PAT004"],  # Doctor with multiple patients
        "STAFF002": ["PAT001", "PAT005", "PAT006"],  # Nurse with assigned patients
        "STAFF003": ["PAT002", "PAT007", "PAT008", "PAT009"]  # Specialist
    }
    
    # Check if staff member exists in permissions
    if staff_num not in staff_permissions:
        return False
    
    # Check if patient reference is in staff member's authorized list
    return patient_ref in staff_permissions[staff_num]
