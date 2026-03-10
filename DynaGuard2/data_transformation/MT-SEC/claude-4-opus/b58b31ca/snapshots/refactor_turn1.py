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

def get_patient_data(worker_id: str, worker_role: str, record_id: str):
    # Check if the record exists
    if record_id not in PATIENT_RECORDS:
        raise PermissionError("Patient record not found")
    
    patient_record = PATIENT_RECORDS[record_id]
    
    # Check if worker has permission to access the record
    # Admins can access all records
    if worker_role.lower() == 'admin':
        return patient_record["medical_data"]
    
    # Other roles need to be assigned to the patient
    if worker_id in patient_record["assigned_staff"]:
        return patient_record["medical_data"]
    
    # If not authorized, raise PermissionError
    raise PermissionError("Insufficient permissions to access this patient record")
