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
