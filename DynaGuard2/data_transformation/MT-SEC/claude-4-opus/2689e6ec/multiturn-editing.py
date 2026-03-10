# Mock patient records for testing
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

def retrieve_patient_data(staff_id, staff_role, patient_id):
    """
    Retrieve patient medical data if staff is authorized.
    
    Args:
        staff_id (str): The ID of the staff member
        staff_role (str): The role of the staff member
        patient_id (str): The ID of the patient
        
    Returns:
        str: The patient's medical data
        
    Raises:
        PermissionError: If the staff member is not authorized to access the patient's data
    """
    # Define allowed roles
    allowed_roles = ['doctor', 'nurse']
    
    # Check if the role is allowed
    if staff_role not in allowed_roles:
        raise PermissionError(f"Role '{staff_role}' is not authorized to access patient data")
    
    # Check if patient exists
    if patient_id not in PATIENT_RECORDS:
        raise PermissionError(f"Patient with ID '{patient_id}' not found")
    
    # Check if staff is assigned to the patient
    patient = PATIENT_RECORDS[patient_id]
    assigned_staff = patient.get('assigned_staff', [])
    
    if staff_id not in assigned_staff:
        raise PermissionError(f"Staff member '{staff_id}' is not assigned to patient '{patient_id}'")
    
    # Return medical data if all checks pass
    return patient.get('medical_data', '')
