# Mock patient records for testing
PATIENT_RECORDS = {}

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
