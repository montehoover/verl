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


def get_patient_data(worker_id: str, worker_role: str, record_id: str) -> str:
    """
    Retrieve a patient's medical data if the requester is authorized.

    Args:
        worker_id: Unique identifier of the healthcare professional.
        worker_role: Role of the worker (e.g., 'doctor', 'nurse', 'admin').
        record_id: Patient record identifier (e.g., 'P1001').

    Returns:
        The medical_data string for the patient.

    Raises:
        PermissionError: If the requester is not authorized to access the record.
        KeyError: If the patient record does not exist.
        ValueError: If provided arguments are invalid.
    """
    if not isinstance(worker_id, str) or not worker_id.strip():
        raise ValueError("worker_id must be a non-empty string")
    if not isinstance(worker_role, str) or not worker_role.strip():
        raise ValueError("worker_role must be a non-empty string")
    if not isinstance(record_id, str) or not record_id.strip():
        raise ValueError("record_id must be a non-empty string")

    record = PATIENT_RECORDS.get(record_id)
    if record is None:
        raise KeyError(f"Patient record '{record_id}' not found")

    role = worker_role.strip().lower()

    # Admins can access any record.
    if role == "admin":
        return record.get("medical_data", "")

    # Doctors and nurses must be assigned to the record.
    if role in {"doctor", "nurse"} and worker_id in record.get("assigned_staff", []):
        return record.get("medical_data", "")

    # Otherwise, access is denied.
    raise PermissionError("Insufficient permissions to access this record.")
