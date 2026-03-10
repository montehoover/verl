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

# Define allowed roles for accessing patient records
ALLOWED_ROLES_FOR_ACCESS = {'doctor', 'nurse', 'admin'}

def get_patient_record(personnel_id: str, personnel_role: str, patient_identifier: str) -> str:
    """
    Retrieves patient medical data if access is authorized.

    Args:
        personnel_id: The ID of the healthcare personnel.
        personnel_role: The role of the healthcare personnel.
        patient_identifier: The ID of the patient.

    Returns:
        The medical data for the patient if access is authorized,
        otherwise an error message for unauthorized access.
    """
    if personnel_role not in ALLOWED_ROLES_FOR_ACCESS:
        return f"Error: Role '{personnel_role}' is not authorized to access patient records."

    patient_record = PATIENT_RECORDS.get(patient_identifier)

    if not patient_record:
        return f"Error: Patient record for '{patient_identifier}' not found."

    if personnel_id not in patient_record["assigned_staff"]:
        return f"Error: Personnel '{personnel_id}' is not assigned to patient '{patient_identifier}'."

    # If all checks pass, return the medical data
    return patient_record["medical_data"]

if __name__ == '__main__':
    # Example Usage:
    # Scenario 1: Authorized access by a doctor
    print(f"Accessing P1001 by D101 (doctor): {get_patient_record('D101', 'doctor', 'P1001')}")
    # Expected: Medical history for patient P1001

    # Scenario 2: Authorized access by a nurse
    print(f"Accessing P1002 by N202 (nurse): {get_patient_record('N202', 'nurse', 'P1002')}")
    # Expected: Medical history for patient P1002

    # Scenario 3: Unauthorized access due to unassigned staff
    print(f"Accessing P1001 by D102 (doctor): {get_patient_record('D102', 'doctor', 'P1001')}")
    # Expected: Error: Personnel 'D102' is not assigned to patient 'P1001'.

    # Scenario 4: Unauthorized access due to invalid role
    print(f"Accessing P1001 by D101 (receptionist): {get_patient_record('D101', 'receptionist', 'P1001')}")
    # Expected: Error: Role 'receptionist' is not authorized to access patient records.

    # Scenario 5: Attempt to access a non-existent patient record
    print(f"Accessing P1003 by D101 (doctor): {get_patient_record('D101', 'doctor', 'P1003')}")
    # Expected: Error: Patient record for 'P1003' not found.

    # Scenario 6: Admin access (assuming admin is assigned or has universal access based on logic)
    # For this example, admin still needs to be in 'assigned_staff'
    PATIENT_RECORDS["P1001"]["assigned_staff"].append("A001") # Temporarily assign an admin
    print(f"Accessing P1001 by A001 (admin): {get_patient_record('A001', 'admin', 'P1001')}")
    # Expected: Medical history for patient P1001
