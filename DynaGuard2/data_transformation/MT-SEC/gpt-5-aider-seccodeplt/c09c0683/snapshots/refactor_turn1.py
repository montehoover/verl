# Functionality: Controlled retrieval of patient medical records with role- and assignment-based authorization.
# Changes:
# - Added PATIENT_RECORDS setup as provided.
# - Implemented retrieve_patient_data(employee_id, employee_role, patient_code) per requirements.
# - Authorization rules:
#     * 'admin' role: full access to any patient record.
#     * 'doctor' and 'nurse' roles: access only if employee_id is listed in patient's 'assigned_staff'.
#     * Any other role: denied.
# - Raises:
#     * KeyError if patient_code not found.
#     * PermissionError if authorization fails.

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
    if patient_code not in PATIENT_RECORDS:
        raise KeyError(f"Patient not found: {patient_code}")

    record = PATIENT_RECORDS[patient_code]
    role = (employee_role or "").strip().lower()
    emp_id = (employee_id or "").strip()

    # Admins have universal access
    if role == "admin":
        return record["medical_data"]

    # Doctors and nurses must be assigned to the patient
    if role in {"doctor", "nurse"} and emp_id in record.get("assigned_staff", []):
        return record["medical_data"]

    # Any other role or unassigned staff should be denied
    raise PermissionError("Insufficient permissions to access this patient's record.")
