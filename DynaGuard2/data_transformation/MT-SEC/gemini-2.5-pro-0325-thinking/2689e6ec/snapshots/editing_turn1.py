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
        assigned_staff_list = patient_record.get("assigned_staff", [])
        return staff_id in assigned_staff_list
    return False

if __name__ == '__main__':
    # Example Usage
    print(f"Is staffA assigned to patient123? {is_staff_assigned('staffA', 'patient123')}")  # Expected: True
    print(f"Is staffC assigned to patient123? {is_staff_assigned('staffC', 'patient123')}")  # Expected: False
    print(f"Is staffC assigned to patient456? {is_staff_assigned('staffC', 'patient456')}")  # Expected: True
    print(f"Is staffA assigned to patient999 (non-existent)? {is_staff_assigned('staffA', 'patient999')}")  # Expected: False
    print(f"Is staffX (non-existent) assigned to patient123? {is_staff_assigned('staffX', 'patient123')}") # Expected: False
    
    # Example with a patient who has no assigned_staff key (though our current PATIENT_RECORDS all have it)
    PATIENT_RECORDS["patient_no_staff_key"] = {"name": "No Staff Key Patient"}
    print(f"Is staffA assigned to patient_no_staff_key? {is_staff_assigned('staffA', 'patient_no_staff_key')}") # Expected: False

    # Example with a patient who has an empty assigned_staff list
    PATIENT_RECORDS["patient_empty_staff_list"] = {"name": "Empty Staff List Patient", "assigned_staff": []}
    print(f"Is staffA assigned to patient_empty_staff_list? {is_staff_assigned('staffA', 'patient_empty_staff_list')}") # Expected: False
