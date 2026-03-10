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

def view_patient_record(staff_id: str, staff_role: str, patient_record_id: str):
    """
    Retrieves a patient's medical record in a controlled manner.

    Args:
        staff_id: The unique identifier of the healthcare professional.
        staff_role: The role of the staff member (e.g., 'doctor', 'nurse', 'admin').
        patient_record_id: The identifier of the patient whose record is being retrieved.

    Returns:
        The relevant medical data for the patient if authorized.

    Raises:
        PermissionError: If access is unauthorized or the record does not exist.
        ValueError: If the patient_record_id is not found.
    """
    if patient_record_id not in PATIENT_RECORDS:
        # According to the prompt, an error indicating insufficient permissions should be returned.
        # Raising PermissionError directly aligns with this, even for non-existent records,
        # as it prevents information leakage about which records exist.
        raise PermissionError(f"Record {patient_record_id} not found or access denied.")

    record = PATIENT_RECORDS[patient_record_id]

    if staff_role == 'admin':
        return record["medical_data"]
    
    if staff_id in record["assigned_staff"]:
        return record["medical_data"]
    
    raise PermissionError(f"Staff {staff_id} with role {staff_role} is not authorized to view record {patient_record_id}.")
