# Mock database for patient records
PATIENT_RECORDS = {}

# Role-based access control configuration
ROLE_PERMISSIONS = {
    'doctor': ['view', 'edit', 'prescribe'],
    'nurse': ['view', 'edit'],
    'admin': ['view', 'edit', 'delete', 'assign']
}

def is_staff_assigned(staff_id, patient_id):
    """
    Check if a staff member is assigned to a specific patient.
    
    Args:
        staff_id (str): The ID of the staff member
        patient_id (str): The ID of the patient
        
    Returns:
        bool: True if the staff member is assigned to the patient, False otherwise
    """
    if patient_id in PATIENT_RECORDS:
        patient_record = PATIENT_RECORDS[patient_id]
        if 'assigned_staff' in patient_record:
            return staff_id in patient_record['assigned_staff']
    return False

def has_access(staff_id, role, patient_id):
    """
    Check if a staff member has role-based access to a patient's record.
    
    Args:
        staff_id (str): The ID of the staff member
        role (str): The role of the staff member (e.g., 'doctor', 'nurse', 'admin')
        patient_id (str): The ID of the patient
        
    Returns:
        bool: True if the staff member has access based on their role, False otherwise
    """
    # Check if the role exists in the permissions system
    if role not in ROLE_PERMISSIONS:
        return False
    
    # Check if patient exists
    if patient_id not in PATIENT_RECORDS:
        return False
    
    # Admins have access to all patient records
    if role == 'admin':
        return True
    
    # For other roles, check if they are assigned to the patient
    return is_staff_assigned(staff_id, patient_id)

def get_patient_data(staff_num, role, patient_ref):
    """
    Retrieve a patient's medical record if the staff member is authorized.
    
    Args:
        staff_num (str): The ID of the healthcare professional requesting access
        role (str): The role of the healthcare worker
        patient_ref (str): The identifier of the patient whose record is being accessed
        
    Returns:
        str: The patient's medical data if authorized
        
    Raises:
        PermissionError: If the staff member is not authorized to access the patient's record
        ValueError: If the patient record does not exist
    """
    # Check if patient exists
    if patient_ref not in PATIENT_RECORDS:
        raise ValueError(f"Patient record for {patient_ref} not found")
    
    # Check if staff member has access
    if not has_access(staff_num, role, patient_ref):
        raise PermissionError(f"Staff member {staff_num} with role '{role}' is not authorized to access patient {patient_ref}")
    
    # Return the medical data
    return PATIENT_RECORDS[patient_ref]['medical_data']
