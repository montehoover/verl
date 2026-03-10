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

def retrieve_patient_data(staff_id: str, staff_role: str, patient_id: str) -> str:
    """
    Facilitates healthcare staff in retrieving a patient's medical record.
    
    Args:
        staff_id: ID of the healthcare professional requesting access
        staff_role: Role of the healthcare worker ('doctor', 'nurse', 'admin', etc.)
        patient_id: Identifier of the patient whose record is being accessed
        
    Returns:
        The patient's medical data if authorized
        
    Raises:
        PermissionError: For unauthorized access
    """
    # Check if patient exists
    if patient_id not in PATIENT_RECORDS:
        raise PermissionError("Patient record not found")
    
    patient_record = PATIENT_RECORDS[patient_id]
    
    # Check authorization based on role
    if staff_role == 'admin':
        # Admins can access any patient record
        return patient_record['medical_data']
    elif staff_role in ['doctor', 'nurse']:
        # Doctors and nurses can only access assigned patients
        if staff_id in patient_record['assigned_staff']:
            return patient_record['medical_data']
        else:
            raise PermissionError(f"Staff member {staff_id} is not authorized to access patient {patient_id}'s records")
    else:
        # Other roles cannot access medical data
        raise PermissionError(f"Role '{staff_role}' is not authorized to access medical records")
