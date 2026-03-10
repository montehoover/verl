# Sample PATIENT_RECORDS dictionary (replace with your actual data structure)
PATIENT_RECORDS = {
    "patient123": {
        "name": "John Doe",
        "assigned_staff": ["staffA", "staffB", "staffC"]
    },
    "patient456": {
        "name": "Jane Smith",
        "assigned_staff": ["staffB", "staffD"]
    },
    "patient789": {
        "name": "Peter Jones",
        "assigned_staff": []  # No staff assigned
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

if __name__ == '__main__':
    # Example Usage for is_staff_assigned:
    print(f"Is staffA assigned to patient123? {is_staff_assigned('staffA', 'patient123')}")  # Expected: True
    print(f"Is staffD assigned to patient123? {is_staff_assigned('staffD', 'patient123')}")  # Expected: False
    print(f"Is staffB assigned to patient456? {is_staff_assigned('staffB', 'patient456')}")  # Expected: True
    print(f"Is staffA assigned to patient789? {is_staff_assigned('staffA', 'patient789')}")  # Expected: False
    print(f"Is staffA assigned to non_existent_patient? {is_staff_assigned('staffA', 'non_existent_patient')}") # Expected: False

    # Example Usage for has_access:
    print(f"\n--- Access Control ---")
    # staffA is a doctor assigned to patient123
    print(f"staffA (doctor) access patient123? {has_access('staffA', 'doctor', 'patient123')}")  # Expected: True
    # staffD is a doctor NOT assigned to patient123
    print(f"staffD (doctor) access patient123? {has_access('staffD', 'doctor', 'patient123')}")  # Expected: False
    # staffB is a nurse assigned to patient456
    print(f"staffB (nurse) access patient456? {has_access('staffB', 'nurse', 'patient456')}")    # Expected: True
    # staffC is an admin (can access any patient, even if not directly assigned)
    print(f"staffC (admin) access patient789? {has_access('staffC', 'admin', 'patient789')}")    # Expected: True
    print(f"staffC (admin) access patient123? {has_access('staffC', 'admin', 'patient123')}")    # Expected: True
    # staffE is a receptionist (not a defined role with special access)
    print(f"staffE (receptionist) access patient123? {has_access('staffE', 'receptionist', 'patient123')}") # Expected: False
    # Accessing a non-existent patient record
    print(f"staffA (doctor) access non_existent_patient? {has_access('staffA', 'doctor', 'non_existent_patient')}") # Expected: False
