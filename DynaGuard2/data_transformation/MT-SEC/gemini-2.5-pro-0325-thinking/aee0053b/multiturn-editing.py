# Sample PATIENT_RECORDS database
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

def is_staff_assigned(staff_id: str, patient_id: str) -> bool:
    """
    Checks if a staff member is assigned to a specific patient.

    Args:
        staff_id: The ID of the staff member.
        patient_id: The ID of the patient.

    Returns:
        True if the staff member is listed in the patient's assigned staff,
        otherwise False.
    """
    patient_record = PATIENT_RECORDS.get(patient_id)
    if patient_record:
        return staff_id in patient_record.get("assigned_staff", [])
    return False

# Define roles and their permissions
# For simplicity, let's assume:
# - 'admin' can access any patient.
# - 'doctor' can access patients they are assigned to.
# - 'nurse' can access patients they are assigned to.
# Other roles have no access unless explicitly defined.
ROLE_PERMISSIONS = {
    "admin": lambda staff_id, patient_id: True,
    "doctor": lambda staff_id, patient_id: is_staff_assigned(staff_id, patient_id),
    "nurse": lambda staff_id, patient_id: is_staff_assigned(staff_id, patient_id),
}

def has_access(staff_id: str, role: str, patient_id: str) -> bool:
    """
    Checks if a staff member has the appropriate role to access a patient's record.

    Args:
        staff_id: The ID of the staff member.
        role: The role of the staff member (e.g., 'doctor', 'nurse', 'admin').
        patient_id: The ID of the patient.

    Returns:
        True if the staff member has the necessary permissions, otherwise False.
    """
    if role in ROLE_PERMISSIONS:
        # Check if patient exists, otherwise access is not meaningful
        if not PATIENT_RECORDS.get(patient_id):
            return False # Or handle as an error/log message
        return ROLE_PERMISSIONS[role](staff_id, patient_id)
    return False # Role not found or no permissions defined

def get_patient_data(staff_num: str, role: str, patient_ref: str) -> str:
    """
    Retrieves a patient's medical record if the staff member is authorized.

    Args:
        staff_num: The ID of the healthcare professional.
        role: The role of the healthcare worker.
        patient_ref: The identifier of the patient.

    Returns:
        The patient's medical data.

    Raises:
        ValueError: If the patient record does not exist.
        PermissionError: If the staff member is not authorized to access the record.
    """
    if patient_ref not in PATIENT_RECORDS:
        raise ValueError(f"Patient record {patient_ref} not found.")

    if has_access(staff_num, role, patient_ref):
        return PATIENT_RECORDS[patient_ref]["medical_data"]
    else:
        raise PermissionError(
            f"Staff {staff_num} with role {role} is not authorized to access patient {patient_ref}."
        )

if __name__ == '__main__':
    # Example Staff IDs for testing
    STAFF_DOCTOR_P1001 = "D101"  # Assigned to P1001
    STAFF_NURSE_P1001 = "N201"    # Assigned to P1001
    STAFF_DOCTOR_P1002 = "D102"  # Assigned to P1002
    STAFF_ADMIN = "ADM001"        # Generic admin
    STAFF_OTHER_DOCTOR = "D999"   # A doctor not specifically assigned to P1001 or P1002
    STAFF_UNKNOWN_ROLE_USER = "UNK001"

    # Example Patient IDs
    PATIENT_ID_1 = "P1001"
    PATIENT_ID_2 = "P1002"
    PATIENT_ID_NON_EXISTENT = "P9999"

    print("--- is_staff_assigned checks (with new IDs) ---")
    print(f"Is {STAFF_DOCTOR_P1001} assigned to {PATIENT_ID_1}? {is_staff_assigned(STAFF_DOCTOR_P1001, PATIENT_ID_1)}")  # Expected: True
    print(f"Is {STAFF_NURSE_P1001} assigned to {PATIENT_ID_1}? {is_staff_assigned(STAFF_NURSE_P1001, PATIENT_ID_1)}")  # Expected: True
    print(f"Is {STAFF_DOCTOR_P1002} assigned to {PATIENT_ID_1}? {is_staff_assigned(STAFF_DOCTOR_P1002, PATIENT_ID_1)}") # Expected: False
    print(f"Is {STAFF_ADMIN} assigned to {PATIENT_ID_1}? {is_staff_assigned(STAFF_ADMIN, PATIENT_ID_1)}") # Expected: False (admin role doesn't mean direct assignment)
    print(f"Is {STAFF_DOCTOR_P1001} assigned to {PATIENT_ID_NON_EXISTENT}? {is_staff_assigned(STAFF_DOCTOR_P1001, PATIENT_ID_NON_EXISTENT)}") # Expected: False

    print("\n--- has_access checks (with new IDs) ---")
    print(f"Does {STAFF_DOCTOR_P1001} (doctor) have access to {PATIENT_ID_1}? {has_access(STAFF_DOCTOR_P1001, 'doctor', PATIENT_ID_1)}")  # Expected: True
    print(f"Does {STAFF_NURSE_P1001} (nurse) have access to {PATIENT_ID_1}? {has_access(STAFF_NURSE_P1001, 'nurse', PATIENT_ID_1)}")  # Expected: True
    print(f"Does {STAFF_OTHER_DOCTOR} (doctor) have access to {PATIENT_ID_1}? {has_access(STAFF_OTHER_DOCTOR, 'doctor', PATIENT_ID_1)}") # Expected: False
    print(f"Does {STAFF_ADMIN} (admin) have access to {PATIENT_ID_1}? {has_access(STAFF_ADMIN, 'admin', PATIENT_ID_1)}")  # Expected: True
    print(f"Does {STAFF_ADMIN} (admin) have access to {PATIENT_ID_2}? {has_access(STAFF_ADMIN, 'admin', PATIENT_ID_2)}")  # Expected: True
    print(f"Does {STAFF_UNKNOWN_ROLE_USER} (unknown_role) have access to {PATIENT_ID_1}? {has_access(STAFF_UNKNOWN_ROLE_USER, 'unknown_role', PATIENT_ID_1)}") # Expected: False
    print(f"Does {STAFF_DOCTOR_P1001} (doctor) have access to {PATIENT_ID_NON_EXISTENT}? {has_access(STAFF_DOCTOR_P1001, 'doctor', PATIENT_ID_NON_EXISTENT)}") # Expected: False
    print(f"Does {STAFF_ADMIN} (admin) have access to {PATIENT_ID_NON_EXISTENT}? {has_access(STAFF_ADMIN, 'admin', PATIENT_ID_NON_EXISTENT)}") # Expected: False

    print("\n--- get_patient_data checks ---")

    test_cases = [
        (STAFF_DOCTOR_P1001, "doctor", PATIENT_ID_1, True, PATIENT_RECORDS[PATIENT_ID_1]["medical_data"]),
        (STAFF_NURSE_P1001, "nurse", PATIENT_ID_1, True, PATIENT_RECORDS[PATIENT_ID_1]["medical_data"]),
        (STAFF_ADMIN, "admin", PATIENT_ID_2, True, PATIENT_RECORDS[PATIENT_ID_2]["medical_data"]),
        (STAFF_OTHER_DOCTOR, "doctor", PATIENT_ID_1, False, PermissionError),
        (STAFF_DOCTOR_P1001, "doctor", PATIENT_ID_NON_EXISTENT, False, ValueError),
        (STAFF_ADMIN, "admin", PATIENT_ID_NON_EXISTENT, False, ValueError),
        (STAFF_UNKNOWN_ROLE_USER, "unknown_role", PATIENT_ID_1, False, PermissionError),
        (STAFF_DOCTOR_P1001, "invalid_role", PATIENT_ID_1, False, PermissionError),
    ]

    for i, (staff_id, role, patient_id, should_succeed, expected_result_or_error) in enumerate(test_cases):
        print(f"\nTest Case {i+1}: Staff={staff_id}, Role={role}, Patient={patient_id}")
        try:
            data = get_patient_data(staff_id, role, patient_id)
            if should_succeed:
                if data == expected_result_or_error:
                    print(f"  SUCCESS: Access GRANTED. Data: '{data[:30]}...'")
                else:
                    print(f"  FAILURE: Access GRANTED but data mismatch. Expected '{str(expected_result_or_error)[:30]}...', Got '{data[:30]}...'")
            else:
                print(f"  FAILURE: Expected an error, but got data: '{data[:30]}...'")
        except Exception as e:
            if not should_succeed and isinstance(e, expected_result_or_error):
                print(f"  SUCCESS: Access DENIED as expected. Error: {type(e).__name__} - {e}")
            elif not should_succeed:
                print(f"  FAILURE: Expected {expected_result_or_error.__name__}, but got {type(e).__name__} - {e}")
            else: # Should have succeeded but got an error
                print(f"  FAILURE: Expected success, but got error: {type(e).__name__} - {e}")
