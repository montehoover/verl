# Mock patient records for testing
PATIENT_RECORDS = {}

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
