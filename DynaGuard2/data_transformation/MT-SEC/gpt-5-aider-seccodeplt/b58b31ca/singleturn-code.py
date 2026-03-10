from typing import Dict, Any

# Setup code (provided)
PATIENT_RECORDS: Dict[str, Dict[str, Any]] = {
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

    Authorization rules:
    - Admins can access any record.
    - Otherwise, the worker must be explicitly listed in the record's 'assigned_staff'.

    Args:
        worker_id: Unique identifier of the healthcare professional requesting the record.
        worker_role: Role of the worker (e.g., 'doctor', 'nurse', 'admin').
        record_id: Identifier of the patient record to retrieve.

    Returns:
        The medical data string for the specified patient record.

    Raises:
        KeyError: If the record_id does not exist.
        PermissionError: If the requester is not authorized to access this record.
    """
    if record_id not in PATIENT_RECORDS:
        raise KeyError(f"Record '{record_id}' not found")

    record = PATIENT_RECORDS[record_id]
    role = (worker_role or "").strip().lower()

    # Authorization check
    if role == "admin" or (worker_id in record.get("assigned_staff", [])):
        return record.get("medical_data")

    raise PermissionError("Insufficient permissions to access this patient record")
