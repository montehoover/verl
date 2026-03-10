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

def get_patient_data(staff_num: str, role: str, patient_ref: str):
    # Check if patient exists
    if patient_ref not in PATIENT_RECORDS:
        raise PermissionError("Unauthorized access")
    
    patient_record = PATIENT_RECORDS[patient_ref]
    
    # Check authorization based on role
    if role == 'admin':
        # Admins have access to all records
        return patient_record['medical_data']
    elif role in ['doctor', 'nurse']:
        # Doctors and nurses need to be assigned to the patient
        if staff_num in patient_record['assigned_staff']:
            return patient_record['medical_data']
        else:
            raise PermissionError("Unauthorized access")
    else:
        # Other roles (receptionist, etc.) don't have access to medical data
        raise PermissionError("Unauthorized access")
