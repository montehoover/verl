from typing import Dict, Any

# Setup: patient records database
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


def view_patient_record(staff_id: str, staff_role: str, patient_record_id: str) -> str:
    """
    Retrieve a patient's medical record in a controlled manner.

    Authorization policy:
    - Access granted if:
      - staff_role is 'admin' (case-insensitive), OR
      - staff_id is listed in the patient's assigned_staff.
    - Otherwise, raise PermissionError.

    Args:
        staff_id: Unique identifier of the requesting healthcare professional.
        staff_role: Role of the staff member (e.g., 'doctor', 'nurse', 'admin').
        patient_record_id: Identifier of the patient record to retrieve.

    Returns:
        The patient's medical_data string if authorized.

    Raises:
        ValueError: If any required argument is missing/invalid.
        KeyError: If the patient record does not exist.
        PermissionError: If the requester is not authorized to view the record.
    """
    # Basic validation
    if not isinstance(staff_id, str) or not staff_id.strip():
        raise ValueError("staff_id must be a non-empty string.")
    if not isinstance(staff_role, str) or not staff_role.strip():
        raise ValueError("staff_role must be a non-empty string.")
    if not isinstance(patient_record_id, str) or not patient_record_id.strip():
        raise ValueError("patient_record_id must be a non-empty string.")

    normalized_role = staff_role.strip().lower()
    pid = patient_record_id.strip()

    # Fetch the patient record
    try:
        record = PATIENT_RECORDS[pid]
    except KeyError as e:
        raise KeyError(f"Patient record '{pid}' not found.") from e

    # Authorization check
    is_admin = normalized_role == "admin"
    is_assigned = staff_id.strip() in record.get("assigned_staff", [])

    if not (is_admin or is_assigned):
        raise PermissionError("Insufficient permissions to view this patient record.")

    # Return the medical data if authorized
    return record.get("medical_data", "")
