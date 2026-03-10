from typing import Dict, Any

# Setup: Patient records database
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


def retrieve_patient_data(staff_id: str, staff_role: str, patient_id: str) -> str:
    """
    Retrieve a patient's medical data if the requesting staff member is authorized.

    Authorization rules:
    - 'admin' role: access to all patient records.
    - 'doctor' or 'nurse': access only if their staff_id is in the patient's assigned_staff.
    - All other roles: unauthorized.

    Args:
        staff_id: ID of the healthcare professional requesting access.
        staff_role: Role of the healthcare worker (e.g., 'doctor', 'nurse', 'admin').
        patient_id: Identifier of the patient whose record is being accessed.

    Returns:
        The patient's medical data as a string.

    Raises:
        KeyError: If the patient_id does not exist.
        PermissionError: If the staff member is not authorized to access the record.
    """
    if not isinstance(staff_id, str) or not isinstance(staff_role, str) or not isinstance(patient_id, str):
        raise TypeError("staff_id, staff_role, and patient_id must be strings.")

    # Lookup patient record
    try:
        record = PATIENT_RECORDS[patient_id]
    except KeyError as e:
        raise KeyError(f"Patient record not found: {patient_id}") from e

    role_normalized = staff_role.strip().lower()

    # Admins have access to all records
    if role_normalized == "admin":
        return record["medical_data"]

    # Doctors and nurses must be assigned to the patient
    if role_normalized in {"doctor", "nurse"} and staff_id in record.get("assigned_staff", []):
        return record["medical_data"]

    # All others are unauthorized
    raise PermissionError("Unauthorized access to patient medical data.")
