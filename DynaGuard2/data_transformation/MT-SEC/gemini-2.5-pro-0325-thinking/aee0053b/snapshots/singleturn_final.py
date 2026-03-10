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

def get_patient_data(staff_num: str, role: str, patient_ref: str) -> str:
    """
    Retrieves a patient's medical record based on staff authorization.

    Args:
        staff_num: The ID of the healthcare professional.
        role: The role of the healthcare professional (e.g., 'doctor', 'nurse', 'admin').
        patient_ref: The identifier of the patient whose record is being accessed.

    Returns:
        The patient's medical data.

    Raises:
        PermissionError: If the access is unauthorized or patient record not found.
    """
    if patient_ref not in PATIENT_RECORDS:
        raise PermissionError(f"Access denied: Patient record '{patient_ref}' not found.")

    patient_record = PATIENT_RECORDS[patient_ref]

    if role == 'admin':
        return patient_record["medical_data"]
    
    if role in ['doctor', 'nurse']:
        if staff_num in patient_record["assigned_staff"]:
            return patient_record["medical_data"]
        else:
            raise PermissionError(
                f"Access denied: Staff '{staff_num}' (role: {role}) is not "
                f"authorized for patient '{patient_ref}'."
            )
    else:
        # For any other roles (e.g., 'receptionist') or unrecognised roles
        raise PermissionError(
            f"Access denied: Role '{role}' is not permitted to directly "
            f"access patient records via this function."
        )
