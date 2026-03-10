# Mock patient records for testing
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

# Role-based access permissions
ROLE_PERMISSIONS = {
    'doctor': ['view', 'edit', 'prescribe'],
    'nurse': ['view', 'edit'],
    'admin': ['view', 'edit', 'delete', 'assign_staff']
}

def is_staff_assigned(staff_id, patient_record_id):
    """
    Check if a staff member is assigned to a specific patient.
    
    Args:
        staff_id (str): The ID of the staff member
        patient_record_id (str): The ID of the patient record
        
    Returns:
        bool: True if the staff member is assigned to the patient, False otherwise
    """
    if patient_record_id not in PATIENT_RECORDS:
        return False
    
    patient_record = PATIENT_RECORDS[patient_record_id]
    assigned_staff = patient_record.get('assigned_staff', [])
    
    return staff_id in assigned_staff

def has_access(staff_id, role, patient_record_id):
    """
    Check if a staff member has role-based access to a patient record.
    
    Args:
        staff_id (str): The ID of the staff member
        role (str): The role of the staff member ('doctor', 'nurse', or 'admin')
        patient_record_id (str): The ID of the patient record
        
    Returns:
        bool: True if the staff member has access based on their role, False otherwise
    """
    # Check if the role exists in permissions
    if role not in ROLE_PERMISSIONS:
        return False
    
    # Admins have access to all patient records
    if role == 'admin':
        return True
    
    # For doctors and nurses, check if they are assigned to the patient
    return is_staff_assigned(staff_id, patient_record_id)

def view_patient_record(staff_id, staff_role, patient_record_id):
    """
    Retrieve a patient's medical record if the staff member is authorized.
    
    Args:
        staff_id (str): The unique identifier of the healthcare professional
        staff_role (str): The role of the staff member
        patient_record_id (str): The identifier of the patient
        
    Returns:
        dict or str: The patient's medical data if authorized, otherwise an error message
    """
    # Check if the patient record exists
    if patient_record_id not in PATIENT_RECORDS:
        return "Error: Patient record not found"
    
    # Check if the staff member has access
    if not has_access(staff_id, staff_role, patient_record_id):
        return "Error: Insufficient permissions to access this patient record"
    
    # Return the medical data
    patient_record = PATIENT_RECORDS[patient_record_id]
    return {
        "patient_id": patient_record["patient_id"],
        "medical_data": patient_record["medical_data"]
    }
