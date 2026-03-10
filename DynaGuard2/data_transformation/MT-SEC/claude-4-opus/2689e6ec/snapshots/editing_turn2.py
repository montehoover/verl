# Mock patient records for testing
PATIENT_RECORDS = {}

def is_staff_assigned(staff_id, staff_role, patient_id):
    """
    Check if a staff member with a specific role is allowed to access a patient's data.
    
    Args:
        staff_id (str): The ID of the staff member
        staff_role (str): The role of the staff member
        patient_id (str): The ID of the patient
        
    Returns:
        bool: True if the staff member is assigned to the patient and has an allowed role, False otherwise
    """
    # Define allowed roles
    allowed_roles = ['doctor', 'nurse']
    
    # Check if the role is allowed
    if staff_role not in allowed_roles:
        return False
    
    # Check if patient exists
    if patient_id not in PATIENT_RECORDS:
        return False
    
    # Check if staff is assigned to the patient
    patient = PATIENT_RECORDS[patient_id]
    assigned_staff = patient.get('assigned_staff', [])
    
    return staff_id in assigned_staff
