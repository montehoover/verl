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

    if worker_role == 'admin':
        return patient_record["medical_data"]
    
    if worker_id in patient_record["assigned_staff"]:
        return patient_record["medical_data"]
    
    raise PermissionError(f"Worker {worker_id} with role {worker_role} has insufficient permissions to access record {record_id}.")
