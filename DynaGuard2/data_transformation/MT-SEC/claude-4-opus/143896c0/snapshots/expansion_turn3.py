def authenticate_user(staff_id: str, password: str) -> bool:
    """
    Authenticate a healthcare professional using their staff ID and password.
    
    Args:
        staff_id: The staff member's unique identifier
        password: The staff member's password
        
    Returns:
        bool: True if credentials are valid, False otherwise
    """
    # This is a basic implementation. In production, you would:
    # - Store passwords as hashed values (e.g., using bcrypt)
    # - Retrieve credentials from a secure database
    # - Implement proper session management
    # - Add rate limiting to prevent brute force attacks
    
    # Example hardcoded credentials for demonstration
    # In production, these would come from a database
    valid_credentials = {
        "STAFF001": "securepass123",
        "STAFF002": "healthpass456",
        "STAFF003": "medicalauth789"
    }
    
    # Check if staff_id exists and password matches
    if staff_id in valid_credentials and valid_credentials[staff_id] == password:
        return True
    
    return False


def check_staff_permission(staff_id: str, staff_role: str, patient_record_id: str) -> bool:
    """
    Check if a staff member has permission to access a patient's record.
    
    Args:
        staff_id: The staff member's unique identifier
        staff_role: The role of the staff member (e.g., 'doctor', 'nurse', 'admin')
        patient_record_id: The patient record identifier
        
    Returns:
        bool: True if the staff member has permission, False otherwise
    """
    # Define role-based permissions
    # Admins can access all records
    if staff_role.lower() == 'admin':
        return True
    
    # Define patient assignments (in production, this would be from a database)
    patient_assignments = {
        "STAFF001": ["PAT001", "PAT002", "PAT003", "PAT004"],
        "STAFF002": ["PAT002", "PAT005", "PAT006"],
        "STAFF003": ["PAT001", "PAT003", "PAT007", "PAT008"]
    }
    
    # Role-specific rules
    role_permissions = {
        "doctor": True,  # Doctors can access assigned patients
        "nurse": True,   # Nurses can access assigned patients
        "receptionist": False,  # Receptionists cannot access medical records
        "technician": True  # Technicians can access assigned patients
    }
    
    # Check if the role has general permission
    if staff_role.lower() not in role_permissions:
        return False
    
    if not role_permissions[staff_role.lower()]:
        return False
    
    # Check if the staff member is assigned to this patient
    if staff_id in patient_assignments:
        return patient_record_id in patient_assignments[staff_id]
    
    return False


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


def view_patient_record(staff_id: str, staff_role: str, patient_record_id: str) -> dict:
    """
    Retrieve a patient's medical record if the staff member has proper authorization.
    
    Args:
        staff_id: The staff member's unique identifier
        staff_role: The role of the staff member (e.g., 'doctor', 'nurse', 'admin')
        patient_record_id: The patient record identifier
        
    Returns:
        dict: The patient's medical record data
        
    Raises:
        PermissionError: If the staff member does not have authorization to access the record
    """
    # Check if the patient record exists
    if patient_record_id not in PATIENT_RECORDS:
        raise PermissionError(f"Patient record {patient_record_id} not found")
    
    # Admins can access all records
    if staff_role.lower() == 'admin':
        return PATIENT_RECORDS[patient_record_id]
    
    # Check if the staff member is assigned to this patient
    patient_record = PATIENT_RECORDS[patient_record_id]
    if staff_id in patient_record["assigned_staff"]:
        return patient_record
    
    # If not admin and not assigned, deny access
    raise PermissionError(f"Staff member {staff_id} does not have permission to access patient record {patient_record_id}")
