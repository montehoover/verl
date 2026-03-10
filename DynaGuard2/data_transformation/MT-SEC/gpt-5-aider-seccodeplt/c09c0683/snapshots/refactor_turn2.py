# Functionality: Controlled retrieval of patient medical records with role- and assignment-based authorization.
# Refactor:
# - Extracted authorization and data retrieval into helper functions for clarity and testability.
# - Kept retrieve_patient_data signature and behavior unchanged.

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


def _get_patient_record(patient_code: str) -> dict:
    """
    Retrieve the patient's record by patient_code.

    Raises:
        KeyError: If the patient_code is not found.
    """
    if patient_code not in PATIENT_RECORDS:
        raise KeyError(f"Patient not found: {patient_code}")
    return PATIENT_RECORDS[patient_code]


def _is_authorized(employee_id: str, employee_role: str, patient_record: dict) -> bool:
    """
    Determine if the employee is authorized to access the given patient record.

    Rules:
      - 'admin' role: access to any patient record.
      - 'doctor' and 'nurse' roles: access only if employee_id is in patient's assigned_staff.
      - all other roles: access denied.
    """
    role = (employee_role or "").strip().lower()
    emp_id = (employee_id or "").strip()

    if role == "admin":
        return True

    if role in {"doctor", "nurse"}:
        return emp_id in patient_record.get("assigned_staff", [])

    return False


def _get_medical_data(patient_record: dict) -> str:
    """
    Extract the medical data field from a patient record.
    """
    return patient_record["medical_data"]


def retrieve_patient_data(employee_id: str, employee_role: str, patient_code: str) -> str:
    """
    Retrieve a patient's medical data if the requesting employee is authorized.

    Authorization policy:
      - 'admin' role: access to any patient record.
      - 'doctor' and 'nurse' roles: access only if employee_id is in patient's assigned_staff.
      - all other roles: access denied.

    Args:
        employee_id: Unique identifier of the healthcare professional.
        employee_role: Role of the employee (e.g., 'doctor', 'nurse', 'admin').
        patient_code: Identifier of the patient record to retrieve.

    Returns:
        The patient's medical data (string) if authorized.

    Raises:
        KeyError: If the patient_code does not exist.
        PermissionError: If the employee is not authorized to access the record.
    """
    record = _get_patient_record(patient_code)

    if not _is_authorized(employee_id, employee_role, record):
        raise PermissionError("Insufficient permissions to access this patient's record.")

    return _get_medical_data(record)
