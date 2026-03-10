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
    # For this example, we'll use a placeholder implementation.

    if staff_role == "admin":
        return True  # Admins have access to all patient data

    if staff_role == "doctor":
        # Doctors might only have access to patients assigned to them.
        # This is a simplified check; a real system would query a database.
        # Let's assume doctors can access patients with IDs starting with "P" followed by their own ID.
        # e.g., doctor "doc123" can access "Pdoc123_001", "Pdoc123_002"
        if patient_id.startswith(f"P{staff_id}_"):
            return True
        # Or, a general rule for doctors to access any patient for now
        # For a more robust example, you'd have a list of patients assigned to a doctor.
        # if patient_id in get_assigned_patients(staff_id):
        #     return True

    if staff_role == "nurse":
        # Nurses might have similar or more restricted access than doctors.
        # Let's assume nurses can access patients whose IDs contain their staff_id as a substring.
        # This is a simplified placeholder.
        if staff_id in patient_id:
            return True

    return False
