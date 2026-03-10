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

def get_patient_record(personnel_id: str, personnel_role: str, patient_identifier: str):
    """
    Allows healthcare staff to review patient medical data.

    Args:
        personnel_id: The ID of the healthcare personnel trying to access the record.
        personnel_role: The role of the user (e.g., 'doctor', 'nurse', 'admin').
        patient_identifier: The ID of the patient whose records are being sought.

    Returns:
        The medical data for the patient if access is authorized.

    Raises:
        PermissionError: If access is unauthorized.
        ValueError: If the patient_identifier is not found.
    """
    if patient_identifier not in PATIENT_RECORDS:
        raise ValueError(f"Patient record not found for ID: {patient_identifier}")

    patient_record = PATIENT_RECORDS[patient_identifier]

    if personnel_role == 'admin':
        return patient_record["medical_data"]
    
    if personnel_role in ['doctor', 'nurse']:
        if personnel_id in patient_record["assigned_staff"]:
            return patient_record["medical_data"]
        else:
            raise PermissionError(f"Personnel {personnel_id} not authorized for patient {patient_identifier}")
    
    # If role is not admin, doctor, or nurse, or if other conditions weren't met.
    raise PermissionError(f"Role {personnel_role} is not authorized to access patient records.")
