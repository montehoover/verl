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
    Authenticates healthcare staff based on their ID and password.

    Args:
        staff_id: The staff member's unique identifier.
        password: The staff member's password.

    Returns:
        True if the credentials are valid, False otherwise.
    """
    # In a real application, this would involve checking credentials
    # against a secure database or an authentication service.
    # For this example, we'll use a placeholder implementation.
    if staff_id == "admin" and password == "password123":
        return True
    return False

def check_staff_permission(staff_num: str, patient_ref: str) -> bool:
    """
    Checks if a staff member has permission to access a specific patient's record.

    Args:
        staff_num: The staff member's unique identifier.
        patient_ref: The patient's unique reference identifier.

    Returns:
        True if the staff member is authorized, False otherwise.
    """
    # In a real application, this would involve checking against a
    # permissions database or access control list (ACL).
    # This placeholder implementation assumes:
    # - Staff 'doctor_smith' can access patient 'patient_123'.
    # - Staff 'nurse_jones' can access patient 'patient_456'.
    # - All other combinations are unauthorized.
    if staff_num == "doctor_smith" and patient_ref == "patient_123":
        return True
    if staff_num == "nurse_jones" and patient_ref == "patient_456":
        return True
    return False

def get_patient_data(staff_num: str, role: str, patient_ref: str) -> str:
    """
    Retrieves a patient's medical record if the staff member is authorized.

    Args:
        staff_num: The staff member's unique identifier.
        role: The role of the staff member (e.g., 'doctor', 'nurse').
               (Note: role is not used in the current permission logic but included as per request)
        patient_ref: The patient's unique reference identifier.

    Returns:
        The patient's medical data.

    Raises:
        PermissionError: If the staff member is not authorized to access the record.
        KeyError: If the patient_ref does not exist in PATIENT_RECORDS.
    """
    if patient_ref not in PATIENT_RECORDS:
        raise KeyError(f"Patient record {patient_ref} not found.")

    patient_record = PATIENT_RECORDS[patient_ref]
    if staff_num in patient_record["assigned_staff"]:
        return patient_record["medical_data"]
    else:
        raise PermissionError(f"Staff {staff_num} is not authorized to access patient {patient_ref}'s data.")
