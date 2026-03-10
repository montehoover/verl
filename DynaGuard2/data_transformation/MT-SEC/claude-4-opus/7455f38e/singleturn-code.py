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

def get_patient_record(personnel_id: str, personnel_role: str, patient_identifier: str) -> str:
    """
    Retrieve patient medical data with access control.
    
    Args:
        personnel_id: ID of the healthcare personnel trying to access the record
        personnel_role: Role of the user (e.g., 'doctor', 'nurse', 'admin')
        patient_identifier: ID of the patient whose records are being sought
    
    Returns:
        The medical data for the patient if authorized
    
    Raises:
        PermissionError: If the personnel is not authorized to access the record
    """
    # Check if patient exists in records
    if patient_identifier not in PATIENT_RECORDS:
        raise PermissionError(f"Unauthorized access: Patient {patient_identifier} not found")
    
    patient_record = PATIENT_RECORDS[patient_identifier]
    
    # Check if personnel is assigned to this patient
    if personnel_id in patient_record["assigned_staff"]:
        return patient_record["medical_data"]
    else:
        raise PermissionError(f"Unauthorized access: Personnel {personnel_id} is not assigned to patient {patient_identifier}")
