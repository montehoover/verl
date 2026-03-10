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
    Retrieve a patient's medical data with role-based authorization.

    Rules:
    - 'admin': access any patient.
    - 'doctor' or 'nurse': access only if the staff_num is assigned to the patient.
    - other roles (e.g., 'receptionist'): unauthorized.

    Raises:
        KeyError: if the patient_ref does not exist.
        PermissionError: if the requester is not authorized.
    """
    if patient_ref not in PATIENT_RECORDS:
        raise KeyError(f"Patient '{patient_ref}' not found")

    record = PATIENT_RECORDS[patient_ref]
    role_normalized = role.strip().lower()

    if role_normalized == "admin":
        return record["medical_data"]

    if role_normalized in ("doctor", "nurse"):
        if staff_num in record.get("assigned_staff", []):
            return record["medical_data"]
        raise PermissionError(
            f"Access denied: staff member '{staff_num}' is not assigned to patient '{patient_ref}'."
        )

    raise PermissionError(
        f"Access denied: role '{role}' does not have permission to access patient records."
    )
