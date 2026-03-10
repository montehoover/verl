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

def is_staff_assigned(staff_id: str, patient_record_id: str) -> bool:
    """
    Checks if a staff member is assigned to a specific patient.

    Args:
        staff_id: The ID of the staff member.
        patient_record_id: The ID of the patient record.

    Returns:
        True if the staff member is listed in the patient's assigned staff,
        otherwise False.
    """
    patient_record = PATIENT_RECORDS.get(patient_record_id)
    if patient_record:
        return staff_id in patient_record.get("assigned_staff", [])
    return False

def has_access(staff_id: str, role: str, patient_record_id: str) -> bool:
    """
    Checks if a staff member has the appropriate role to access a patient's record.

    Args:
        staff_id: The ID of the staff member.
        role: The role of the staff member (e.g., 'doctor', 'nurse', 'admin').
        patient_record_id: The ID of the patient record.

    Returns:
        True if the staff member has access, otherwise False.
    """
    # Admins have access to all records
    if role == "admin":
        return True

    # Doctors and nurses can only access records of patients they are assigned to
    if role in ["doctor", "nurse"]:
        if is_staff_assigned(staff_id, patient_record_id):
            return True
    
    # Default to no access
    return False

def view_patient_record(staff_id: str, staff_role: str, patient_record_id: str) -> str:
    """
    Retrieves a patient's medical record if the staff member is authorized.

    Args:
        staff_id: The unique identifier of the healthcare professional.
        staff_role: The role of the staff member.
        patient_record_id: The identifier of the patient.

    Returns:
        The patient's medical data if authorized, or an error message.
    """
    patient_record = PATIENT_RECORDS.get(patient_record_id)
    if not patient_record:
        return "Error: Patient record not found."

    if has_access(staff_id, staff_role, patient_record_id):
        return patient_record.get("medical_data", "Error: Medical data not found for this patient.")
    else:
        return "Error: Insufficient permissions to view this patient record."

if __name__ == '__main__':
    print("--- Patient Record Access Tests ---")

    # Test case 1: Doctor D101 (assigned) accessing patient P1001
    print(f"D101 (doctor) viewing P1001: {view_patient_record('D101', 'doctor', 'P1001')}")
    # Expected: Medical history for patient P1001

    # Test case 2: Nurse N201 (assigned) accessing patient P1001
    print(f"N201 (nurse) viewing P1001: {view_patient_record('N201', 'nurse', 'P1001')}")
    # Expected: Medical history for patient P1001

    # Test case 3: Doctor D101 (not assigned to P1002) accessing patient P1002
    print(f"D101 (doctor) viewing P1002: {view_patient_record('D101', 'doctor', 'P1002')}")
    # Expected: Error: Insufficient permissions to view this patient record.

    # Test case 4: Admin A001 accessing patient P1001 (Admins have universal access)
    print(f"A001 (admin) viewing P1001: {view_patient_record('A001', 'admin', 'P1001')}")
    # Expected: Medical history for patient P1001

    # Test case 5: Admin A001 accessing patient P1002
    print(f"A001 (admin) viewing P1002: {view_patient_record('A001', 'admin', 'P1002')}")
    # Expected: Medical history for patient P1002

    # Test case 6: Staff R301 (receptionist role, not in assigned_staff) accessing P1001
    print(f"R301 (receptionist) viewing P1001: {view_patient_record('R301', 'receptionist', 'P1001')}")
    # Expected: Error: Insufficient permissions to view this patient record.

    # Test case 7: Accessing a non-existent patient record P9999
    print(f"D101 (doctor) viewing P9999: {view_patient_record('D101', 'doctor', 'P9999')}")
    # Expected: Error: Patient record not found.

    # Test case 8: Doctor D102 (assigned) accessing patient P1002
    print(f"D102 (doctor) viewing P1002: {view_patient_record('D102', 'doctor', 'P1002')}")
    # Expected: Medical history for patient P1002

    print("\n--- Direct function tests with new data (optional verification) ---")
    print(f"Is D101 assigned to P1001? {is_staff_assigned('D101', 'P1001')}") # Expected: True
    print(f"Is D101 assigned to P1002? {is_staff_assigned('D101', 'P1002')}") # Expected: False
    print(f"D101 (doctor) has access to P1001? {has_access('D101', 'doctor', 'P1001')}") # Expected: True
    print(f"D101 (doctor) has access to P1002? {has_access('D101', 'doctor', 'P1002')}") # Expected: False
    print(f"A001 (admin) has access to P1001? {has_access('A001', 'admin', 'P1001')}") # Expected: True
