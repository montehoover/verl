# Assume PATIENT_RECORDS is available in the setup.
# Example structure:
# PATIENT_RECORDS = {
#     "patient123": {
#         "name": "John Doe",
#         "assigned_staff": ["staffA", "staffB"]
#     },
#     "patient456": {
#         "name": "Jane Smith",
#         "assigned_staff": ["staffC"]
#     }
# }

# For the purpose of this example, let's define it here.
PATIENT_RECORDS = {
    "patient123": {
        "name": "John Doe",
        "dob": "1980-01-01",
        "assigned_staff": ["staffA", "staffB", "staffD"]
    },
    "patient456": {
        "name": "Jane Smith",
        "dob": "1992-05-15",
        "assigned_staff": ["staffC"]
    },
    "patient789": {
        "name": "Alice Brown",
        "dob": "1975-11-30",
        "assigned_staff": ["staffA", "staffE"]
    }
}

ALLOWED_ROLES_FOR_ACCESS = ["doctor", "nurse"]

def can_staff_access_patient_data(staff_id: str, staff_role: str, patient_id: str) -> bool:
    """
    Checks if a staff member is assigned to a specific patient and has an allowed role.

    Args:
        staff_id: The ID of the staff member.
        staff_role: The role of the staff member.
        patient_id: The ID of the patient.

    Returns:
        True if the staff member is assigned to the patient and has an allowed role,
        otherwise False.
    """
    if staff_role.lower() not in ALLOWED_ROLES_FOR_ACCESS:
        return False

    patient_record = PATIENT_RECORDS.get(patient_id)
    if patient_record:
        assigned_staff_list = patient_record.get("assigned_staff", [])
        return staff_id in assigned_staff_list
    return False

if __name__ == '__main__':
    # Example Usage
    print(f"Can staffA (doctor) access patient123? {can_staff_access_patient_data('staffA', 'doctor', 'patient123')}")  # Expected: True
    print(f"Can staffB (nurse) access patient123? {can_staff_access_patient_data('staffB', 'nurse', 'patient123')}")  # Expected: True
    print(f"Can staffD (admin) access patient123? {can_staff_access_patient_data('staffD', 'admin', 'patient123')}")  # Expected: False (staffD is assigned, but admin role is not allowed)
    print(f"Can staffC (doctor) access patient123? {can_staff_access_patient_data('staffC', 'doctor', 'patient123')}")  # Expected: False (staffC not assigned to patient123)
    print(f"Can staffC (doctor) access patient456? {can_staff_access_patient_data('staffC', 'doctor', 'patient456')}")  # Expected: True
    print(f"Can staffA (doctor) access patient999 (non-existent)? {can_staff_access_patient_data('staffA', 'doctor', 'patient999')}")  # Expected: False
    print(f"Can staffX (non-existent staff) (doctor) access patient123? {can_staff_access_patient_data('staffX', 'doctor', 'patient123')}") # Expected: False
    
    # Example with a patient who has no assigned_staff key
    PATIENT_RECORDS["patient_no_staff_key"] = {"name": "No Staff Key Patient"}
    print(f"Can staffA (doctor) access patient_no_staff_key? {can_staff_access_patient_data('staffA', 'doctor', 'patient_no_staff_key')}") # Expected: False

    # Example with a patient who has an empty assigned_staff list
    PATIENT_RECORDS["patient_empty_staff_list"] = {"name": "Empty Staff List Patient", "assigned_staff": []}
    print(f"Can staffA (doctor) access patient_empty_staff_list? {can_staff_access_patient_data('staffA', 'doctor', 'patient_empty_staff_list')}") # Expected: False

    # Example with a role not in ALLOWED_ROLES_FOR_ACCESS
    print(f"Can staffA (receptionist) access patient123? {can_staff_access_patient_data('staffA', 'receptionist', 'patient123')}") # Expected: False
