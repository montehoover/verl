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

def validate_record_exists(patient_record_id: str) -> None:
    """Validate that the patient record exists."""
    if patient_record_id not in PATIENT_RECORDS:
        raise PermissionError("Patient record not found")

def check_authorization(staff_id: str, staff_role: str, patient_record: dict) -> bool:
    """Check if the staff member is authorized to access the patient record."""
    # Admins can access all records
    if staff_role.lower() == 'admin':
        return True
    
    # Other staff can only access records they are assigned to
    return staff_id in patient_record['assigned_staff']

def retrieve_medical_data(patient_record: dict) -> str:
    """Retrieve the medical data from the patient record."""
    return patient_record['medical_data']

def get_patient_record(patient_record_id: str) -> dict:
    """Get the patient record from the database."""
    return PATIENT_RECORDS[patient_record_id]

def view_patient_record(staff_id: str, staff_role: str, patient_record_id: str):
    # Pipeline pattern: validate -> get record -> check auth -> retrieve data
    validate_record_exists(patient_record_id)
    
    patient_record = get_patient_record(patient_record_id)
    
    if not check_authorization(staff_id, staff_role, patient_record):
        raise PermissionError("Insufficient permissions to access this patient record")
    
    return retrieve_medical_data(patient_record)
