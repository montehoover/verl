# Mock PATIENT_RECORDS for testing purposes
PATIENT_RECORDS = {
    "P1001": {
        "patient_id": "P1001",
        "assigned_staff": ["D101", "N201"],
        "medical_data": "Medical history for patient P1001"
    },
    "P1002": {
        "patient_id": "P1002",
        "assigned_staff": ["D102", "N202"],
        "medical_data": "Medical history for patient P1002"
    }
}

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


def get_patient_data(worker_id, worker_role, record_id):
    """
    Retrieve a patient's medical record with authorization check.
    
    Args:
        worker_id (str): The unique identifier of the healthcare professional
        worker_role (str): The role of the worker
        record_id (str): The identifier of the patient record
        
    Returns:
        str: The medical data if authorized, error message if not
    """
    if has_access_to_record(worker_id, worker_role, record_id):
        if record_id in PATIENT_RECORDS:
            return PATIENT_RECORDS[record_id].get('medical_data', 'No medical data available')
        else:
            return 'Error: Patient record not found'
    else:
        return 'Error: Insufficient permissions to access this record'
