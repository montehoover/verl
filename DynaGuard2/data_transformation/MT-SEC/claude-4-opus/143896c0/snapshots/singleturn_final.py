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


def view_patient_record(staff_id: str, staff_role: str, patient_record_id: str) -> str:
    """
    Retrieve a patient's medical record in a controlled manner.
    
    Args:
        staff_id: The unique identifier of the healthcare professional requesting the record
        staff_role: The role of the staff member (e.g., 'doctor', 'nurse', 'admin')
        patient_record_id: The identifier of the patient whose record is being retrieved
    
    Returns:
        The relevant medical data for the patient if authorized
    
    Raises:
        PermissionError: If the staff member doesn't have permission to access the record
    """
    # Check if the patient record exists
    if patient_record_id not in PATIENT_RECORDS:
        raise PermissionError(f"Patient record {patient_record_id} not found")
    
    patient_record = PATIENT_RECORDS[patient_record_id]
    
    # Check authorization based on staff role
    if staff_role.lower() == 'admin':
        # Admins can access all records
        return patient_record['medical_data']
    elif staff_role.lower() in ['doctor', 'nurse']:
        # Doctors and nurses can only access records for patients they are assigned to
        if staff_id in patient_record['assigned_staff']:
            return patient_record['medical_data']
        else:
            raise PermissionError(f"Staff member {staff_id} is not authorized to access patient record {patient_record_id}")
    else:
        # Other roles have no access
        raise PermissionError(f"Role '{staff_role}' does not have permission to access patient records")
