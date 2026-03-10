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

def get_patient_data(staff_num: str, role: str, patient_ref: str) -> str:
    """
    Retrieve a patient's medical record based on staff authorization.
    
    Args:
        staff_num: ID of the healthcare professional requesting access
        role: Role of the healthcare worker ('doctor', 'nurse', 'admin', etc.)
        patient_ref: Identifier of the patient whose record is being accessed
    
    Returns:
        The patient's medical data if authorized
    
    Raises:
        PermissionError: If the user is not authorized to access the record
    """
    # Check if patient exists
    if patient_ref not in PATIENT_RECORDS:
        raise PermissionError("Unauthorized access")
    
    patient_record = PATIENT_RECORDS[patient_ref]
    
    # Admin role has access to all records
    if role == 'admin':
        return patient_record['medical_data']
    
    # For other roles (doctor, nurse), check if they are assigned to the patient
    if role in ['doctor', 'nurse']:
        if staff_num in patient_record['assigned_staff']:
            return patient_record['medical_data']
    
    # All other cases are unauthorized
    raise PermissionError("Unauthorized access")
