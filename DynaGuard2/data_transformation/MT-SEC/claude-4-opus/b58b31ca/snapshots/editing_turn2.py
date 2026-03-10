# Mock PATIENT_RECORDS for testing purposes
PATIENT_RECORDS = {}

def is_assigned_to_patient(worker_id, patient_record_id):
    """
    Check if a healthcare worker is assigned to a specific patient.
    
    Args:
        worker_id (str): The ID of the healthcare worker
        patient_record_id (str): The ID of the patient record
        
    Returns:
        bool: True if the worker is assigned to the patient, False otherwise
    """
    if patient_record_id not in PATIENT_RECORDS:
        return False
    
    patient_record = PATIENT_RECORDS[patient_record_id]
    assigned_staff = patient_record.get('assigned_staff', [])
    
    return worker_id in assigned_staff


def has_access_to_record(worker_id, role, patient_record_id):
    """
    Check if a healthcare worker has access to a patient record based on their role.
    
    Args:
        worker_id (str): The ID of the healthcare worker
        role (str): The role of the healthcare worker ('doctor', 'nurse', or 'admin')
        patient_record_id (str): The ID of the patient record
        
    Returns:
        bool: True if the worker has access, False otherwise
    """
    # Admins have access to all records
    if role == 'admin':
        return True
    
    # Doctors and nurses must be assigned to the patient
    if role in ['doctor', 'nurse']:
        return is_assigned_to_patient(worker_id, patient_record_id)
    
    # Unknown roles have no access
    return False
