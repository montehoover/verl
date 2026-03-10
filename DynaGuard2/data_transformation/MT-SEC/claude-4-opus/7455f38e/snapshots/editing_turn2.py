# Assuming PATIENT_RECORDS is defined elsewhere in the application
# If not, it should be imported or passed as a parameter

def has_access_to_patient(personnel_id, role, patient_id):
    """
    Check if a healthcare personnel has access to a specific patient based on their role.
    
    Args:
        personnel_id (str): The ID of the healthcare personnel
        role (str): The role of the personnel ('doctor', 'nurse', or 'admin')
        patient_id (str): The ID of the patient
        
    Returns:
        bool: True if the personnel has the appropriate role and access to the patient, False otherwise
    """
    # PATIENT_RECORDS should be available in the global scope or imported
    global PATIENT_RECORDS
    
    # Check if role is valid
    valid_roles = ['doctor', 'nurse', 'admin']
    if role not in valid_roles:
        return False
    
    # Admins have access to all patients
    if role == 'admin':
        return True
    
    # For doctors and nurses, check if they are assigned to the patient
    if patient_id not in PATIENT_RECORDS:
        return False
    
    patient_record = PATIENT_RECORDS[patient_id]
    
    # Check if the patient record has assigned personnel
    if 'assigned_personnel' in patient_record:
        return personnel_id in patient_record['assigned_personnel']
    
    return False
