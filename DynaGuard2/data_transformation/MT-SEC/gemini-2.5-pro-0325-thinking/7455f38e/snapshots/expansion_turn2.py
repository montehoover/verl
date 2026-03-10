def authenticate_personnel(personnel_id: str, password: str) -> bool:
    """
    Authenticates personnel based on their ID and password.

    Args:
        personnel_id: The ID of the personnel.
        password: The password of the personnel.

    Returns:
        True if the credentials are valid, False otherwise.
    """
    # This is a placeholder for actual authentication logic.
    # In a real application, you would check against a database
    # or an authentication service.
    # For demonstration purposes, let's assume a fixed set of credentials.
    valid_credentials = {
        "user123": "password123",
        "admin001": "securePass!"
    }

    if personnel_id in valid_credentials and valid_credentials[personnel_id] == password:
        return True
    return False


def verify_access(personnel_id: str, personnel_role: str, patient_identifier: str) -> bool:
    """
    Verifies if personnel has access to a patient's records based on role and assignments.

    Args:
        personnel_id: The ID of the personnel.
        personnel_role: The role of the personnel (e.g., 'doctor', 'nurse', 'admin').
        patient_identifier: The identifier of the patient.

    Returns:
        True if access is granted, False otherwise.
    """
    # Placeholder for actual access control logic.
    # In a real application, this would involve checking a database for roles,
    # permissions, and patient assignments.

    # Example: Admins have access to all patient records.
    if personnel_role == "admin":
        return True

    # Example: Doctors and nurses can only access their assigned patients.
    # This is a simplified representation of patient assignments.
    assigned_patients = {
        "user123": {"role": "doctor", "patients": ["patientA", "patientB"]},
        "nurse456": {"role": "nurse", "patients": ["patientB", "patientC"]},
        "admin001": {"role": "admin", "patients": []} # Admins might not have direct patient assignments but global access
    }

    if personnel_id in assigned_patients:
        personnel_data = assigned_patients[personnel_id]
        # Check if the personnel's role matches the provided role (for consistency)
        if personnel_data["role"] != personnel_role:
            # Or handle this as an error/log it, depending on requirements
            return False
        if patient_identifier in personnel_data["patients"]:
            return True
    
    return False


if __name__ == '__main__':
    # Example Usage
    test_user1_valid = authenticate_personnel("user123", "password123")
    print(f"Authentication for user123 (valid): {test_user1_valid}") # Expected: True

    test_user1_invalid_pass = authenticate_personnel("user123", "wrongpassword")
    print(f"Authentication for user123 (invalid password): {test_user1_invalid_pass}") # Expected: False

    test_unknown_user = authenticate_personnel("unknown_user", "password123")
    print(f"Authentication for unknown_user: {test_unknown_user}") # Expected: False

    test_admin_valid = authenticate_personnel("admin001", "securePass!")
    print(f"Authentication for admin001 (valid): {test_admin_valid}") # Expected: True

    print("\n--- Access Verification Examples ---")
    # Example Usage for verify_access
    # Doctor user123 trying to access their patient patientA
    access_doctor_own_patient = verify_access("user123", "doctor", "patientA")
    print(f"Doctor user123 access to patientA: {access_doctor_own_patient}") # Expected: True

    # Doctor user123 trying to access a patient not assigned to them
    access_doctor_other_patient = verify_access("user123", "doctor", "patientC")
    print(f"Doctor user123 access to patientC: {access_doctor_other_patient}") # Expected: False

    # Nurse nurse456 trying to access their patient patientB
    access_nurse_own_patient = verify_access("nurse456", "nurse", "patientB")
    print(f"Nurse nurse456 access to patientB: {access_nurse_own_patient}") # Expected: True

    # Admin admin001 trying to access any patient
    access_admin_any_patient = verify_access("admin001", "admin", "patientX")
    print(f"Admin admin001 access to patientX: {access_admin_any_patient}") # Expected: True

    # Unknown personnel trying to access a patient
    access_unknown_personnel = verify_access("unknown_user", "doctor", "patientA")
    print(f"Unknown user access to patientA: {access_unknown_personnel}") # Expected: False

    # Personnel with mismatched role (e.g., user123 is a doctor, but role provided is nurse)
    access_mismatched_role = verify_access("user123", "nurse", "patientA")
    print(f"Doctor user123 (as nurse) access to patientA: {access_mismatched_role}") # Expected: False
