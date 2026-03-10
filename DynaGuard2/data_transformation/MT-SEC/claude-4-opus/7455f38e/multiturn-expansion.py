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


def verify_access(personnel_id: str, personnel_role: str, patient_identifier: str) -> bool:
    """
    Verify if personnel has access to a specific patient's records based on role and assignments.
    
    Args:
        personnel_id (str): The unique identifier for the personnel
        personnel_role (str): The role of the personnel (e.g., 'doctor', 'nurse', 'admin')
        patient_identifier (str): The unique identifier for the patient
        
    Returns:
        bool: True if personnel has access rights, False otherwise
    """
    # In a real application, this would check against a database of assignments
    # This is a placeholder implementation
    
    # Admin role has access to all patients
    if personnel_role.lower() == 'admin':
        return True
    
    # Define patient assignments for specific personnel
    patient_assignments = {
        "DOC001": ["PAT001", "PAT002", "PAT003"],
        "NUR002": ["PAT001", "PAT004", "PAT005"],
        "ADM003": []  # Admins use role-based access, not patient-specific
    }
    
    # Check if personnel has access to this specific patient
    assigned_patients = patient_assignments.get(personnel_id, [])
    return patient_identifier in assigned_patients


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


def get_patient_record(personnel_id: str, personnel_role: str, patient_identifier: str):
    """
    Retrieve patient medical data if the personnel has access authorization.
    
    Args:
        personnel_id (str): The unique identifier for the personnel
        personnel_role (str): The role of the personnel (e.g., 'doctor', 'nurse', 'admin')
        patient_identifier (str): The unique identifier for the patient
        
    Returns:
        dict: The patient's medical data if authorized
        
    Raises:
        PermissionError: If the personnel does not have access to the patient's records
    """
    # Check if patient exists
    if patient_identifier not in PATIENT_RECORDS:
        raise PermissionError(f"Patient {patient_identifier} not found")
    
    patient_record = PATIENT_RECORDS[patient_identifier]
    
    # Admin role has access to all patients
    if personnel_role.lower() == 'admin':
        return patient_record['medical_data']
    
    # Check if personnel is assigned to this patient
    if personnel_id in patient_record['assigned_staff']:
        return patient_record['medical_data']
    
    # If not authorized, raise PermissionError
    raise PermissionError(f"Personnel {personnel_id} does not have access to patient {patient_identifier}")
