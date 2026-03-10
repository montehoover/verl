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
    # Patient identifiers (e.g., "P1001") should align with PATIENT_RECORDS keys.
    assigned_patients = {
        "user123": {"role": "doctor", "patients": ["P1001"]},
        "nurse456": {"role": "nurse", "patients": ["P1002"]},
        "admin001": {"role": "admin", "patients": []} # Admins have global access to all patients conceptually
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


def get_patient_record(personnel_id: str, personnel_role: str, patient_identifier: str) -> str:
    """
    Retrieves patient medical data if personnel is authorized.

    Args:
        personnel_id: The ID of the personnel.
        personnel_role: The role of the personnel.
        patient_identifier: The identifier of the patient.

    Returns:
        The patient's medical data.

    Raises:
        PermissionError: If personnel is not authorized to access the record.
        ValueError: If the patient record does not exist after authorization.
    """
    if not verify_access(personnel_id, personnel_role, patient_identifier):
        raise PermissionError(
            f"Personnel {personnel_id} (role: {personnel_role}) is not authorized to access patient {patient_identifier}."
        )

    if patient_identifier not in PATIENT_RECORDS:
        # This case implies verify_access might grant access to a patient ID
        # that doesn't exist in PATIENT_RECORDS (e.g., admin access to any ID).
        raise ValueError(f"Patient record not found for identifier: {patient_identifier}")

    return PATIENT_RECORDS[patient_identifier]["medical_data"]


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

    print("\n--- Access Verification Examples (Updated Patient IDs) ---")
    # Example Usage for verify_access
    # Doctor user123 trying to access their patient P1001
    access_doctor_own_patient = verify_access("user123", "doctor", "P1001")
    print(f"Doctor user123 access to P1001: {access_doctor_own_patient}") # Expected: True

    # Doctor user123 trying to access a patient not assigned to them (P1002)
    access_doctor_other_patient = verify_access("user123", "doctor", "P1002")
    print(f"Doctor user123 access to P1002: {access_doctor_other_patient}") # Expected: False

    # Nurse nurse456 trying to access their patient P1002
    access_nurse_own_patient = verify_access("nurse456", "nurse", "P1002")
    print(f"Nurse nurse456 access to P1002: {access_nurse_own_patient}") # Expected: True

    # Admin admin001 trying to access any patient (P1001)
    access_admin_any_patient = verify_access("admin001", "admin", "P1001")
    print(f"Admin admin001 access to P1001: {access_admin_any_patient}") # Expected: True
    
    # Admin admin001 trying to access a non-existent patient ID (P9999)
    access_admin_non_existent_patient = verify_access("admin001", "admin", "P9999")
    print(f"Admin admin001 access to P9999: {access_admin_non_existent_patient}") # Expected: True (admin has general access)

    # Unknown personnel trying to access a patient
    access_unknown_personnel = verify_access("unknown_user", "doctor", "P1001")
    print(f"Unknown user access to P1001: {access_unknown_personnel}") # Expected: False

    # Personnel with mismatched role (e.g., user123 is a doctor, but role provided is nurse)
    access_mismatched_role = verify_access("user123", "nurse", "P1001")
    print(f"Doctor user123 (as nurse) access to P1001: {access_mismatched_role}") # Expected: False

    print("\n--- Get Patient Record Examples ---")
    # 1. Doctor "user123" accessing their assigned patient "P1001"
    try:
        record = get_patient_record("user123", "doctor", "P1001")
        print(f"Doctor user123 successfully accessed P1001 data: {record}") # Expected: Medical history for P1001
    except (PermissionError, ValueError) as e:
        print(f"Error for user123 accessing P1001: {e}")

    # 2. Doctor "user123" trying to access patient "P1002" (not assigned)
    try:
        record = get_patient_record("user123", "doctor", "P1002")
        print(f"Doctor user123 accessed P1002 data: {record}")
    except (PermissionError, ValueError) as e:
        print(f"Error for user123 accessing P1002: {e}") # Expected: PermissionError

    # 3. Admin "admin001" accessing patient "P1002"
    try:
        record = get_patient_record("admin001", "admin", "P1002")
        print(f"Admin admin001 successfully accessed P1002 data: {record}") # Expected: Medical history for P1002
    except (PermissionError, ValueError) as e:
        print(f"Error for admin001 accessing P1002: {e}")

    # 4. Admin "admin001" accessing non-existent patient "P9999"
    try:
        record = get_patient_record("admin001", "admin", "P9999")
        print(f"Admin admin001 accessed P9999 data: {record}")
    except (PermissionError, ValueError) as e:
        print(f"Error for admin001 accessing P9999: {e}") # Expected: ValueError

    # 5. Unauthorized personnel "intruder" trying to access "P1001"
    try:
        record = get_patient_record("intruder", "doctor", "P1001")
        print(f"Intruder accessed P1001 data: {record}")
    except (PermissionError, ValueError) as e:
        print(f"Error for intruder accessing P1001: {e}") # Expected: PermissionError
        
    # 6. Nurse "nurse456" accessing their assigned patient "P1002"
    try:
        record = get_patient_record("nurse456", "nurse", "P1002")
        print(f"Nurse nurse456 successfully accessed P1002 data: {record}") # Expected: Medical history for P1002
    except (PermissionError, ValueError) as e:
        print(f"Error for nurse456 accessing P1002: {e}")
