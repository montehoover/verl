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


def authenticate_staff(staff_id: str, password: str) -> bool:
    """
    Authenticates staff based on staff_id and password.

    Args:
        staff_id: The staff member's ID.
        password: The staff member's password.

    Returns:
        True if authentication is successful, False otherwise.
    """
    # In a real application, this would involve:
    # 1. Securely fetching stored credentials (e.g., from a database).
    # 2. Comparing the provided password with a hashed version of the stored password.
    # For this example, we'll use a placeholder implementation.
    if staff_id == "admin" and password == "password123":
        return True
    return False


def check_staff_permission(staff_id: str, staff_role: str, patient_id: str) -> bool:
    """
    Checks if a staff member has permission to access a patient's data.

    Args:
        staff_id: The staff member's ID.
        staff_role: The role of the staff member (e.g., 'doctor', 'nurse', 'admin').
        patient_id: The ID of the patient whose data is being accessed.

    Returns:
        True if the staff member has permission, False otherwise.
    """
    # In a real application, this would involve:
    # 1. Checking against a roles and permissions database.
    # 2. Verifying if the staff member is assigned to the patient (for roles like 'doctor' or 'nurse').
    # This function now uses PATIENT_RECORDS for more realistic permission checks.

    if patient_id not in PATIENT_RECORDS:
        return False # Cannot grant permission for a non-existent patient

    if staff_role == "admin":
        return True  # Admins have access to all patient data

    patient_record = PATIENT_RECORDS[patient_id]
    if staff_role in ["doctor", "nurse"]:
        if staff_id in patient_record["assigned_staff"]:
            return True

    return False


def retrieve_patient_data(staff_id: str, staff_role: str, patient_id: str) -> str:
    """
    Retrieves a patient's medical record if the staff member is authorized.

    Args:
        staff_id: The staff member's ID.
        staff_role: The role of the staff member.
        patient_id: The ID of the patient whose data is to be retrieved.

    Returns:
        The patient's medical data.

    Raises:
        PermissionError: If the staff member is not authorized to access the data.
        ValueError: If the patient_id does not exist in the records.
    """
    if patient_id not in PATIENT_RECORDS:
        raise ValueError(f"Patient with ID '{patient_id}' not found.")

    if not check_staff_permission(staff_id, staff_role, patient_id):
        raise PermissionError(
            f"Staff member '{staff_id}' with role '{staff_role}' is not authorized to access data for patient '{patient_id}'."
        )

    return PATIENT_RECORDS[patient_id]["medical_data"]
