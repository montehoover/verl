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


def _fetch_patient_data(patient_identifier: str) -> dict:
    """
    Retrieves patient data from the PATIENT_RECORDS.

    Args:
        patient_identifier: The ID of the patient.

    Returns:
        The patient's record dictionary.

    Raises:
        ValueError: If the patient_identifier is not found.
    """
    if patient_identifier not in PATIENT_RECORDS:
        raise ValueError(f"Patient record not found for ID: {patient_identifier}")
    return PATIENT_RECORDS[patient_identifier]


def _authorize_access(personnel_id: str, personnel_role: str, patient_record: dict, patient_identifier: str):
    """
    Checks if the personnel has authorization to access the patient record.

    Args:
        personnel_id: The ID of the healthcare personnel.
        personnel_role: The role of the user.
        patient_record: The patient's record dictionary.
        patient_identifier: The ID of the patient (for error messaging).

    Raises:
        PermissionError: If access is unauthorized.
    """
    if personnel_role == 'admin':
        return  # Admin has access

    if personnel_role in ['doctor', 'nurse']:
        if personnel_id in patient_record["assigned_staff"]:
            return  # Assigned staff has access
        else:
            raise PermissionError(f"Personnel {personnel_id} not authorized for patient {patient_identifier}")
    
    # If role is not admin, doctor, or nurse, or if other conditions weren't met.
    raise PermissionError(f"Role {personnel_role} is not authorized to access patient records.")


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
    patient_record = _fetch_patient_data(patient_identifier)
    _authorize_access(personnel_id, personnel_role, patient_record, patient_identifier)
    return patient_record["medical_data"]
