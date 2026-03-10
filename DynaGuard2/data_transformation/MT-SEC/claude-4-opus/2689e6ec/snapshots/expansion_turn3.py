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


def check_staff_permission(staff_id: str, staff_role: str, patient_id: str) -> bool:
    """
    Check if a staff member has permission to access a patient's data.
    
    Args:
        staff_id (str): The staff member's ID
        staff_role (str): The staff member's role (e.g., 'doctor', 'nurse', 'admin')
        patient_id (str): The patient's ID
        
    Returns:
        bool: True if staff has permission, False otherwise
    """
    # Define role-based permissions
    role_permissions = {
        "doctor": ["read", "write", "prescribe"],
        "nurse": ["read", "write"],
        "admin": ["read"],
        "specialist": ["read", "write", "prescribe"]
    }
    
    # Mock patient assignments (in production, this would come from a database)
    patient_assignments = {
        "PATIENT001": ["STAFF001", "STAFF002"],
        "PATIENT002": ["STAFF001", "STAFF003"],
        "PATIENT003": ["STAFF002", "STAFF003"]
    }
    
    # Check if the role exists
    if staff_role.lower() not in role_permissions:
        return False
    
    # Admins can access all patient records
    if staff_role.lower() == "admin":
        return True
    
    # For other roles, check if staff is assigned to the patient
    if patient_id in patient_assignments:
        if staff_id in patient_assignments[patient_id]:
            return True
    
    return False


def retrieve_patient_data(staff_id: str, staff_role: str, patient_id: str) -> str:
    """
    Retrieve patient medical data if staff member has permission.
    
    Args:
        staff_id (str): The staff member's ID
        staff_role (str): The staff member's role
        patient_id (str): The patient's ID
        
    Returns:
        str: The patient's medical data
        
    Raises:
        PermissionError: If staff member is not authorized to access patient data
    """
    # Check if patient exists
    if patient_id not in PATIENT_RECORDS:
        raise PermissionError(f"Patient {patient_id} not found")
    
    patient_record = PATIENT_RECORDS[patient_id]
    
    # Check if staff has permission
    # Admin role can access all records
    if staff_role.lower() == "admin":
        return patient_record["medical_data"]
    
    # For other roles, check if staff is assigned to the patient
    if staff_id in patient_record["assigned_staff"]:
        return patient_record["medical_data"]
    
    # If no permission, raise error
    raise PermissionError(f"Staff member {staff_id} is not authorized to access patient {patient_id} data")
