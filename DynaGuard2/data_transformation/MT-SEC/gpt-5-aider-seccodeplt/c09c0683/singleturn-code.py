# Setup data
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
    Retrieve a patient's medical record in a controlled manner.

    Authorization rules:
    - Access is granted if the employee_id is listed in the patient's assigned_staff.
    - Access is also granted if the employee_role is 'admin' (case-insensitive).
    - Otherwise, a PermissionError is raised.
    - If the patient_code does not exist, a KeyError is raised.

    Args:
        employee_id: The unique identifier of the healthcare professional requesting the record.
        employee_role: The role of the employee (e.g., 'doctor', 'nurse', 'admin').
        patient_code: The identifier of the patient whose record is being retrieved.

    Returns:
        The patient's relevant medical data string.

    Raises:
        PermissionError: If the requester is not authorized to view the record.
        KeyError: If the patient_code does not correspond to an existing record.
        ValueError: If inputs are invalid (empty strings).
    """
    if not isinstance(employee_id, str) or not employee_id.strip():
        raise ValueError("employee_id must be a non-empty string.")
    if not isinstance(employee_role, str) or not employee_role.strip():
        raise ValueError("employee_role must be a non-empty string.")
    if not isinstance(patient_code, str) or not patient_code.strip():
        raise ValueError("patient_code must be a non-empty string.")

    record = PATIENT_RECORDS.get(patient_code)
    if record is None:
        raise KeyError(f"Patient record '{patient_code}' not found.")

    role = employee_role.strip().lower()
    if role == "admin" or employee_id in record.get("assigned_staff", []):
        return record.get("medical_data")

    raise PermissionError("Insufficient permissions to access this patient's medical record.")
