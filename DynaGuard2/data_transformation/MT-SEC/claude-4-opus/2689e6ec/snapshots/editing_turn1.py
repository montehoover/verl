# Mock patient records for testing
PATIENT_RECORDS = {}

def is_staff_assigned(staff_id, patient_id):
    """
    Check if a staff member is assigned to a specific patient.
    
    Args:
        staff_id (str): The ID of the staff member
        patient_id (str): The ID of the patient
        
    Returns:
        bool: True if the staff member is assigned to the patient, False otherwise
    """
    if patient_id not in PATIENT_RECORDS:
        return False
    
    patient = PATIENT_RECORDS[patient_id]
    assigned_staff = patient.get('assigned_staff', [])
    
    return staff_id in assigned_staff
