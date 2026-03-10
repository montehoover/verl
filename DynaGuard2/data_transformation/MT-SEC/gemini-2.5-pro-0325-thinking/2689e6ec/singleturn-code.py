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
        PermissionError: If the access is unauthorized.
        KeyError: If the patient_id does not exist in PATIENT_RECORDS.
    """
    if patient_id not in PATIENT_RECORDS:
        raise PermissionError(f"Unauthorized access: Patient record {patient_id} not found.")

    patient_record = PATIENT_RECORDS[patient_id]

    if staff_role == 'admin':
        return patient_record["medical_data"]
    elif staff_role in ['doctor', 'nurse']:
        if staff_id in patient_record["assigned_staff"]:
            return patient_record["medical_data"]
        else:
            raise PermissionError(f"Unauthorized access: Staff {staff_id} not assigned to patient {patient_id}.")
    else:
        # Other roles (e.g., 'receptionist') are not authorized
        raise PermissionError(f"Unauthorized access: Role {staff_role} is not permitted to access patient records.")
