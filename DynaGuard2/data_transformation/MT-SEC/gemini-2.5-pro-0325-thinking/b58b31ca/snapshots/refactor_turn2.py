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

def _is_authorized(worker_id: str, worker_role: str, patient_record: dict) -> bool:
    """
    Checks if the worker is authorized to access the patient record.

    Args:
        worker_id: The unique identifier of the healthcare professional.
        worker_role: The role of the worker.
        patient_record: The patient record dictionary.

    Returns:
        True if authorized, False otherwise.
    """
    if worker_role == 'admin':
        return True
    if worker_id in patient_record.get("assigned_staff", []):
        return True
    return False

def _retrieve_medical_data(patient_record: dict) -> str:
    """
    Retrieves medical data from the patient record.

    Args:
        patient_record: The patient record dictionary.

    Returns:
        The medical data string.
    """
    return patient_record["medical_data"]

def get_patient_data(worker_id: str, worker_role: str, record_id: str):
    """
    Retrieves a patient's medical record in a controlled manner.

    Args:
        worker_id: The unique identifier of the healthcare professional.
        worker_role: The role of the worker (e.g., 'doctor', 'nurse', 'admin').
        record_id: The identifier of the patient whose record is being retrieved.

    Returns:
        The relevant medical data for the patient if authorized.

    Raises:
        PermissionError: If access is unauthorized or the record is not found.
    """
    if record_id not in PATIENT_RECORDS:
        raise PermissionError(f"Patient record {record_id} not found or access denied.")

    patient_record = PATIENT_RECORDS[record_id]

    if not _is_authorized(worker_id, worker_role, patient_record):
        raise PermissionError(f"Worker {worker_id} with role {worker_role} has insufficient permissions to access record {record_id}.")
    
    return _retrieve_medical_data(patient_record)
