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

def get_patient_data(staff_num: str, role: str, patient_ref: str):
    """
    Retrieves a patient's medical record based on staff authorization.

    Args:
        staff_num: The ID of the healthcare professional.
        role: The role of the healthcare worker (e.g., 'doctor', 'nurse', 'admin').
        patient_ref: The identifier of the patient.

    Returns:
        The patient's medical data if authorized.

    Raises:
        ValueError: If patient_ref is not found in PATIENT_RECORDS.
        PermissionError: If the staff member is not authorized to access the record.
    """
    if patient_ref not in PATIENT_RECORDS:
        raise ValueError(f"Patient record {patient_ref} not found.")

    patient_record = PATIENT_RECORDS[patient_ref]

    if role == 'admin':
        return patient_record["medical_data"]
    elif role in ['doctor', 'nurse']:
        if staff_num in patient_record["assigned_staff"]:
            return patient_record["medical_data"]
        else:
            raise PermissionError(
                f"Staff {staff_num} ({role}) is not authorized to access patient {patient_ref}'s record."
            )
    else:
        raise PermissionError(
            f"Role '{role}' is not authorized to access patient records."
        )
