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

def retrieve_patient_data(staff_id: str, staff_role: str, patient_id: str):
    """
    Facilitates healthcare staff in retrieving a patient's medical record.

    Args:
        staff_id: The ID of the healthcare professional requesting access.
        staff_role: The role of the healthcare worker (e.g., 'doctor', 'nurse', 'admin').
        patient_id: The identifier of the patient whose record is being accessed.

    Returns:
        The patient's medical data if authorized.

    Raises:
        ValueError: If the patient_id is not found in PATIENT_RECORDS.
        PermissionError: If the access is unauthorized.
    """
    if patient_id not in PATIENT_RECORDS:
        raise ValueError(f"Patient record not found for ID: {patient_id}")

    patient_record = PATIENT_RECORDS[patient_id]

    if staff_role == 'admin':
        return patient_record["medical_data"]
    
    if staff_id in patient_record["assigned_staff"]:
        return patient_record["medical_data"]
    
    raise PermissionError(f"Staff member {staff_id} with role {staff_role} is not authorized to access patient {patient_id}'s data.")
