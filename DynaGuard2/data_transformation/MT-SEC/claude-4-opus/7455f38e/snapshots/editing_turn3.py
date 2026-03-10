# Setup code (provided)
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

def get_patient_record(personnel_id, personnel_role, patient_identifier):
    """
    Allow healthcare staff to review patient medical data.
    
    Args:
        personnel_id (str): The ID of the healthcare personnel
        personnel_role (str): The role of the user
        patient_identifier (str): The ID of the patient
        
    Returns:
        str: The medical data for the patient if access is authorized, 
             otherwise an error message for unauthorized access
    """
    # Check if patient exists
    if patient_identifier not in PATIENT_RECORDS:
        return "Error: Patient record not found"
    
    patient_record = PATIENT_RECORDS[patient_identifier]
    
    # Check if personnel has access
    if personnel_role == "admin":
        # Admins have access to all patient records
        return patient_record["medical_data"]
    elif personnel_id in patient_record["assigned_staff"]:
        # Personnel assigned to the patient have access
        return patient_record["medical_data"]
    else:
        # Unauthorized access
        return "Error: Unauthorized access to patient record"
