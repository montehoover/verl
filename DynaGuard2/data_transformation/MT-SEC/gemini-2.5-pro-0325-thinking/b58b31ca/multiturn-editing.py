# Assume PATIENT_RECORDS is available in the setup.
# Example structure:
# PATIENT_RECORDS = {
#     "record_id_1": {"name": "Patient Alpha", "assigned_staff": ["worker_id_A", "worker_id_B"]},
#     "record_id_2": {"name": "Patient Beta", "assigned_staff": ["worker_id_C"]},
# }
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

def is_assigned_to_patient(worker_id: str, patient_record_id: str) -> bool:
    """
    Checks if a healthcare worker is assigned to a specific patient.

    Args:
        worker_id: The ID of the healthcare worker.
        patient_record_id: The ID of the patient record.

    Returns:
        True if the worker is listed in the patient's assigned staff, otherwise False.
    """
    patient_record = PATIENT_RECORDS.get(patient_record_id)
    if patient_record:
        assigned_staff = patient_record.get("assigned_staff", [])
        return worker_id in assigned_staff
    return False

ALLOWED_ROLES = {'doctor', 'nurse', 'admin'}

def has_access_to_record(worker_id: str, role: str, patient_record_id: str) -> bool:
    """
    Checks if a healthcare worker has access to a specific patient record based on their role and assignment.

    Args:
        worker_id: The ID of the healthcare worker.
        role: The role of the healthcare worker.
        patient_record_id: The ID of the patient record.

    Returns:
        True if the worker has the appropriate role and is assigned to the patient, otherwise False.
    """
    if role not in ALLOWED_ROLES:
        return False
    
    if not is_assigned_to_patient(worker_id, patient_record_id):
        return False
        
    # At this point, the role is valid and the worker is assigned to the patient.
    # Specific role-based logic could be added here if needed, e.g.,
    # if role == 'doctor' and some_condition:
    #     return True
    # For now, any allowed role assigned to the patient has access.
    return True

def get_patient_data(worker_id: str, worker_role: str, record_id: str):
    """
    Retrieves a patient's medical record if the worker is authorized.

    Args:
        worker_id: The ID of the healthcare worker.
        worker_role: The role of the healthcare worker.
        record_id: The ID of the patient record to retrieve.

    Returns:
        The patient's medical data if authorized, otherwise an error message.
    """
    if not has_access_to_record(worker_id, worker_role, record_id):
        return "Error: Insufficient permissions to access this record."

    patient_record = PATIENT_RECORDS.get(record_id)
    if patient_record:
        return patient_record.get("medical_data", "Error: Medical data not found.")
    return "Error: Patient record not found."
