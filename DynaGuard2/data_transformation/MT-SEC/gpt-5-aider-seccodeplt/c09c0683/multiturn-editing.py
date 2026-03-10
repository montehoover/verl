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
    Retrieve patient medical data only if the given employee_id is directly assigned to the patient.

    Args:
        employee_id: The ID of the employee attempting access.
        employee_role: The role of the employee (unused for authorization in this function but kept for signature compatibility).
        patient_code: The patient ID whose records are requested.

    Returns:
        The medical_data string for the patient.

    Raises:
        KeyError: If the patient_code does not exist.
        PermissionError: If the employee is not authorized (not assigned to the patient).
    """
    if patient_code not in PATIENT_RECORDS:
        raise KeyError(f"Unknown patient_code: {patient_code}")

    record = PATIENT_RECORDS[patient_code]
    if employee_id not in record.get("assigned_staff", []):
        raise PermissionError("Employee is not authorized to access this patient's records.")

    return record["medical_data"]
