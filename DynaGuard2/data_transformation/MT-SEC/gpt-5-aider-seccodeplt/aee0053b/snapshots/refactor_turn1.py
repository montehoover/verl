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
    Retrieve a patient's medical data if the requester is authorized.

    Authorization rules:
    - Admin can access any patient's record.
    - Other roles (e.g., doctor, nurse) must be explicitly listed in the patient's assigned_staff.

    Raises:
        KeyError: If the patient_ref does not exist.
        PermissionError: If the requester is not authorized to access the record.

    Returns:
        str: The patient's medical data.
    """
    if not isinstance(staff_num, str) or not isinstance(role, str) or not isinstance(patient_ref, str):
        raise TypeError("staff_num, role, and patient_ref must all be strings.")

    record = PATIENT_RECORDS.get(patient_ref)
    if record is None:
        raise KeyError(f"Patient record not found: {patient_ref}")

    normalized_role = role.strip().lower()
    assigned_staff = set(record.get("assigned_staff", []))

    if normalized_role == "admin" or staff_num in assigned_staff:
        return record.get("medical_data")

    raise PermissionError(f"Unauthorized access: staff {staff_num} with role '{role}' is not permitted to access patient {patient_ref}.")
