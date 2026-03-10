# Mock database for patient records
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
    if patient_id in PATIENT_RECORDS:
        patient_record = PATIENT_RECORDS[patient_id]
        if 'assigned_staff' in patient_record:
            return staff_id in patient_record['assigned_staff']
    return False
