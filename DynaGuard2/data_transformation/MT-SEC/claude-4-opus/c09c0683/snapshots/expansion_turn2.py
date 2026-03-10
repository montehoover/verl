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


# Patient records database (in production, this would be in a secure database)
PATIENT_RECORDS = {
    "PAT001": {
        "patient_code": "PAT001",
        "name": "John Doe",
        "date_of_birth": "1985-03-15",
        "blood_type": "O+",
        "allergies": ["penicillin"],
        "medical_history": ["hypertension", "diabetes type 2"],
        "current_medications": ["metformin", "lisinopril"],
        "emergency_contact": "Jane Doe - 555-0123"
    },
    "PAT002": {
        "patient_code": "PAT002",
        "name": "Mary Smith",
        "date_of_birth": "1990-07-22",
        "blood_type": "A-",
        "allergies": [],
        "medical_history": ["asthma"],
        "current_medications": ["albuterol inhaler"],
        "emergency_contact": "Bob Smith - 555-0456"
    },
    "PAT003": {
        "patient_code": "PAT003",
        "name": "Robert Johnson",
        "date_of_birth": "1978-11-30",
        "blood_type": "B+",
        "allergies": ["latex", "shellfish"],
        "medical_history": ["appendectomy 2015"],
        "current_medications": [],
        "emergency_contact": "Sarah Johnson - 555-0789"
    }
}


def get_patient_info(patient_code: str) -> dict:
    """
    Retrieve complete patient record from PATIENT_RECORDS.
    
    Args:
        patient_code: The patient's unique identifier
        
    Returns:
        dict: Complete patient record if found, empty dict if not found
    """
    return PATIENT_RECORDS.get(patient_code, {})
