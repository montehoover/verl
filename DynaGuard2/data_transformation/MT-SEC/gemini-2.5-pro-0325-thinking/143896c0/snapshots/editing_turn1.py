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

if __name__ == '__main__':
    # Example Usage:
    print(f"Is staffA assigned to patient123? {is_staff_assigned('staffA', 'patient123')}")  # Expected: True
    print(f"Is staffD assigned to patient123? {is_staff_assigned('staffD', 'patient123')}")  # Expected: False
    print(f"Is staffB assigned to patient456? {is_staff_assigned('staffB', 'patient456')}")  # Expected: True
    print(f"Is staffA assigned to patient789? {is_staff_assigned('staffA', 'patient789')}")  # Expected: False
    print(f"Is staffA assigned to non_existent_patient? {is_staff_assigned('staffA', 'non_existent_patient')}") # Expected: False
