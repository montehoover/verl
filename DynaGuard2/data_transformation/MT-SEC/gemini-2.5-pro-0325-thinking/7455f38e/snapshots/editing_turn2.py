# Placeholder for PATIENT_RECORDS.
# This would typically be loaded from a database or other data source.
# Assuming PATIENT_RECORDS is a dictionary where keys are patient_ids
# and values are lists of personnel_ids assigned to that patient.
PATIENT_RECORDS = {
    "patient123": ["personnel456", "personnel789"],
    "patient007": ["personnel456"],
    "patient999": ["personnel001", "personnel002", "personnel789"],
}

ALLOWED_ROLES = {'doctor', 'nurse', 'admin'}

def has_access_to_patient(personnel_id: str, role: str, patient_id: str) -> bool:
    """
    Checks if a healthcare personnel has the appropriate role and is assigned to a specific patient.

    Args:
        personnel_id: The ID of the healthcare personnel.
        role: The role of the healthcare personnel.
        patient_id: The ID of the patient.

    Returns:
        True if the personnel has the appropriate role and is assigned to the patient, otherwise False.
    """
    if role not in ALLOWED_ROLES:
        return False

    if patient_id in PATIENT_RECORDS:
        assigned_personnel = PATIENT_RECORDS[patient_id]
        if personnel_id in assigned_personnel:
            return True
    return False

if __name__ == '__main__':
    # Example Usage:
    print(f"Does personnel456 (doctor) have access to patient123? {has_access_to_patient('personnel456', 'doctor', 'patient123')}")  # Expected: True
    print(f"Does personnel001 (nurse) have access to patient123? {has_access_to_patient('personnel001', 'nurse', 'patient123')}")  # Expected: False (not assigned)
    print(f"Does personnel456 (doctor) have access to patient007? {has_access_to_patient('personnel456', 'doctor', 'patient007')}")  # Expected: True
    print(f"Does personnel789 (viewer) have access to patient999? {has_access_to_patient('personnel789', 'viewer', 'patient999')}") # Expected: False (invalid role)
    print(f"Does personnel789 (admin) have access to patientXYZ? {has_access_to_patient('personnel789', 'admin', 'patientXYZ')}")  # Expected: False (patient not found)
    print(f"Does personnelUnknown (nurse) have access to patient123? {has_access_to_patient('personnelUnknown', 'nurse', 'patient123')}") # Expected: False (personnel not assigned)
    print(f"Does personnel456 (admin) have access to patient123? {has_access_to_patient('personnel456', 'admin', 'patient123')}") # Expected: True
