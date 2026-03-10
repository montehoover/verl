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

def retrieve_patient_data(employee_id: str, employee_role: str, patient_code: str):
    """
    Retrieves patient medical data based on whether the employee is assigned to the patient.

    Access to medical_data is granted if the employee_id is listed in the
    patient's 'assigned_staff'. The employee_role is not directly used for this
    access check but is kept as a parameter for potential future use or context.

    Args:
        employee_id: The ID of the employee requesting access.
        employee_role: The role of the employee (e.g., 'doctor', 'nurse', 'admin').
                       Currently not used for access control in this version.
        patient_code: The code of the patient whose records are being requested.

    Returns:
        A string containing the patient's medical data.


    Raises:
        KeyError: If the patient_code does not correspond to any patient record.
        PermissionError: If the employee_id is not in the patient's 'assigned_staff' list.
    """
    if patient_code not in PATIENT_RECORDS:
        raise KeyError(f"Patient with code '{patient_code}' not found.")

    patient_record = PATIENT_RECORDS[patient_code]

    if employee_id not in patient_record.get("assigned_staff", []):
        raise PermissionError(
            f"Employee '{employee_id}' is not authorized to access records for patient '{patient_code}'."
        )

    # If authorized, return the medical data.
    # The role (employee_role) is not used for this specific access check,
    # but it's passed in and could be used for other logic if needed.
    return patient_record["medical_data"]
