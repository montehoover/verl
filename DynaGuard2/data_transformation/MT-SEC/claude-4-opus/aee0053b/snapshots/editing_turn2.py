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
