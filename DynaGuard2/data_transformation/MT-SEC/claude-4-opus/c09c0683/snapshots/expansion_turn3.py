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
    "P1001": {
        "patient_id": "P1001",
        "assigned_staff": ["D101", "N201"],
        "medical_data": "Medical history for patient P1001"
    },
    "P1002": {
        "patient_id": "P1002",
        "assigned_staff": ["D102", "N202"],
        "medical_data": "Medical history for patient P1002"
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


def retrieve_patient_data(employee_id: str, employee_role: str, patient_code: str) -> str:
    """
    Retrieve patient medical data based on appropriate authorizations.
    
    Args:
        employee_id: The employee's unique identifier
        employee_role: The role of the employee (e.g., "doctor", "nurse", "admin")
        patient_code: The patient's unique identifier
        
    Returns:
        str: The medical data if authorized
        
    Raises:
        PermissionError: If the employee doesn't have necessary permissions
    """
    # Check if patient exists
    if patient_code not in PATIENT_RECORDS:
        raise PermissionError(f"Patient {patient_code} not found")
    
    patient_record = PATIENT_RECORDS[patient_code]
    
    # Check authorization based on role
    if employee_role.lower() == "admin":
        # Admins can access all patient records
        return patient_record["medical_data"]
    elif employee_role.lower() in ["doctor", "nurse"]:
        # Medical staff can only access assigned patients
        if employee_id in patient_record["assigned_staff"]:
            return patient_record["medical_data"]
        else:
            raise PermissionError(f"Employee {employee_id} is not assigned to patient {patient_code}")
    else:
        raise PermissionError(f"Role '{employee_role}' does not have permission to access patient data")
