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

def view_patient_record(staff_id: str, staff_role: str, patient_record_id: str):
    # Check if the patient record exists
    if patient_record_id not in PATIENT_RECORDS:
        raise PermissionError("Patient record not found")
    
    patient_record = PATIENT_RECORDS[patient_record_id]
    
    # Check if staff member is authorized
    # Admins can access all records
    if staff_role.lower() == 'admin':
        return patient_record['medical_data']
    
    # Other staff can only access records they are assigned to
    if staff_id in patient_record['assigned_staff']:
        return patient_record['medical_data']
    
    # If none of the above conditions are met, raise PermissionError
    raise PermissionError("Insufficient permissions to access this patient record")
