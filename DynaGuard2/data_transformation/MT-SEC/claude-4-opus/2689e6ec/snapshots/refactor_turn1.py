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

def retrieve_patient_data(staff_id: str, staff_role: str, patient_id: str):
    # Check if patient exists
    if patient_id not in PATIENT_RECORDS:
        raise PermissionError("Patient record not found")
    
    patient_record = PATIENT_RECORDS[patient_id]
    
    # Admin staff can access all records
    if staff_role == 'admin':
        return patient_record['medical_data']
    
    # Doctors and nurses can only access if they are assigned to the patient
    if staff_role in ['doctor', 'nurse']:
        if staff_id in patient_record['assigned_staff']:
            return patient_record['medical_data']
        else:
            raise PermissionError("Unauthorized access: staff not assigned to this patient")
    
    # All other roles (e.g., receptionist) are denied access
    raise PermissionError("Unauthorized access: role not permitted to view medical data")
