# Placeholder for PATIENT_RECORDS.
# This would typically be loaded from a database or other data source.
# Assuming PATIENT_RECORDS is a dictionary where keys are patient_ids
# and values are lists of personnel_ids assigned to that patient.
PATIENT_RECORDS = {
    "patient123": ["personnel456", "personnel789"],
    "patient007": ["personnel456"],
    "patient999": ["personnel001", "personnel002", "personnel789"],
}

def is_assigned_to_patient(personnel_id: str, patient_id: str) -> bool:
    """
    Checks if a healthcare personnel is assigned to a specific patient.

    Args:
        personnel_id: The ID of the healthcare personnel.
        patient_id: The ID of the patient.

    Returns:
        True if the personnel is assigned to the patient, otherwise False.
    """
    if patient_id in PATIENT_RECORDS:
        assigned_personnel = PATIENT_RECORDS[patient_id]
        if personnel_id in assigned_personnel:
            return True
    return False

if __name__ == '__main__':
    # Example Usage:
    print(f"Is personnel456 assigned to patient123? {is_assigned_to_patient('personnel456', 'patient123')}")  # Expected: True
    print(f"Is personnel001 assigned to patient123? {is_assigned_to_patient('personnel001', 'patient123')}")  # Expected: False
    print(f"Is personnel456 assigned to patient007? {is_assigned_to_patient('personnel456', 'patient007')}")  # Expected: True
    print(f"Is personnel789 assigned to patientXYZ? {is_assigned_to_patient('personnel789', 'patientXYZ')}")  # Expected: False (patient not found)
    print(f"Is personnelUnknown assigned to patient123? {is_assigned_to_patient('personnelUnknown', 'patient123')}") # Expected: False (personnel not assigned)
