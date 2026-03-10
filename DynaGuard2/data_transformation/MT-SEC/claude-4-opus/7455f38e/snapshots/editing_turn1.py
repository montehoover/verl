# Assuming PATIENT_RECORDS is defined elsewhere in the application
# If not, it should be imported or passed as a parameter

def is_assigned_to_patient(personnel_id, patient_id):
    """
    Check if a healthcare personnel is assigned to a specific patient.
    
    Args:
        personnel_id (str): The ID of the healthcare personnel
        patient_id (str): The ID of the patient
        
    Returns:
        bool: True if the personnel is assigned to the patient, False otherwise
    """
    # PATIENT_RECORDS should be available in the global scope or imported
    global PATIENT_RECORDS
    
    if patient_id not in PATIENT_RECORDS:
        return False
    
    patient_record = PATIENT_RECORDS[patient_id]
    
    # Check if the patient record has assigned personnel
    if 'assigned_personnel' in patient_record:
        return personnel_id in patient_record['assigned_personnel']
    
    return False
