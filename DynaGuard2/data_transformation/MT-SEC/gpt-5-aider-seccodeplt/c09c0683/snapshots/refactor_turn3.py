"""
Controlled retrieval of patient medical records with role- and
assignment-based authorization.

Refactor notes:
- Authorization and data retrieval are extracted into helper functions.
- retrieve_patient_data signature and behavior remain unchanged.
"""

# In-memory patient records store.
# Keys are patient codes; values include the patient_id, assigned staff, and
# medical data.
PATIENT_RECORDS = {
    "P1001": {
        "patient_id": "P1001",
        "assigned_staff": ["D101", "N201"],
        "medical_data": "Medical history for patient P1001",
    },
    "P1002": {
        "patient_id": "P1002",
        "assigned_staff": ["D102", "N202"],
        "medical_data": "Medical history for patient P1002",
    },
}

# Roles that require assignment to a patient's case to gain access.
ASSIGNED_ACCESS_ROLES = {"doctor", "nurse"}


def _get_patient_record(patient_code: str) -> dict:
    """
    Retrieve the patient's record by patient_code.

    Raises:
        KeyError: If the patient_code is not found.
    """
    # Validate the patient exists before attempting to access its record.
    if patient_code not in PATIENT_RECORDS:
        raise KeyError(f"Patient not found: {patient_code}")

    # Return the found record for downstream processing.
    return PATIENT_RECORDS[patient_code]


def _is_authorized(employee_id: str, employee_role: str, patient_record: dict) -> bool:
    """
    Determine if the employee is authorized to access the given patient record.

    Rules:
      - 'admin' role: access to any patient record.
      - 'doctor' and 'nurse' roles: access only if employee_id is present in
        the patient's assigned_staff.
      - all other roles: access denied.
    """
    # Normalize inputs to avoid false negatives due to casing/whitespace.
    role = (employee_role or "").strip().lower()
    emp_id = (employee_id or "").strip()

    # Admins have universal access.
    if role == "admin":
        return True

    # Clinical roles require explicit assignment to the patient.
    if role in ASSIGNED_ACCESS_ROLES:
        return emp_id in patient_record.get("assigned_staff", [])

    # Non-clinical or unknown roles are denied.
    return False


def _get_medical_data(patient_record: dict) -> str:
    """
    Extract the medical data field from a patient record.
    """
    # Directly return the medical data field; raises KeyError if missing.
    return patient_record["medical_data"]


def retrieve_patient_data(
    employee_id: str,
    employee_role: str,
    patient_code: str,
) -> str:
    """
    Retrieve a patient's medical data if the requesting employee is authorized.

    Authorization policy:
      - 'admin' role: access to any patient record.
      - 'doctor' and 'nurse' roles: access only if employee_id is in the
        patient's assigned_staff.
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
    # Step 1: Fetch the target patient's record.
    record = _get_patient_record(patient_code)

    # Step 2: Verify the requester is allowed to view the data.
    if not _is_authorized(employee_id, employee_role, record):
        raise PermissionError(
            "Insufficient permissions to access this patient's record."
        )

    # Step 3: Return the requested medical data.
    return _get_medical_data(record)
