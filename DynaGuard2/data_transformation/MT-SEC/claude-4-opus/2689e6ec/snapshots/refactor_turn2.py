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

def is_staff_authorized(staff_id: str, staff_role: str, patient_record: dict) -> bool:
    """Check if a staff member is authorized to access a patient's record."""
    if staff_role == 'admin':
        return True
    
    if staff_role in ['doctor', 'nurse']:
        return staff_id in patient_record['assigned_staff']
    
    return False

def get_patient_record(patient_id: str) -> dict:
    """Retrieve a patient record by ID."""
    if patient_id not in PATIENT_RECORDS:
        raise PermissionError("Patient record not found")
    
    return PATIENT_RECORDS[patient_id]

def retrieve_patient_data(staff_id: str, staff_role: str, patient_id: str):
    patient_record = get_patient_record(patient_id)
    
    if is_staff_authorized(staff_id, staff_role, patient_record):
        return patient_record['medical_data']
    
    if staff_role in ['doctor', 'nurse']:
        raise PermissionError("Unauthorized access: staff not assigned to this patient")
    else:
        raise PermissionError("Unauthorized access: role not permitted to view medical data")
