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


def get_patient_record(personnel_id: str, personnel_role: str, patient_identifier: str):
    # Check if patient exists
    if patient_identifier not in PATIENT_RECORDS:
        raise PermissionError("Unauthorized access")
    
    patient_record = PATIENT_RECORDS[patient_identifier]
    
    # Check access control based on role
    if personnel_role == 'admin':
        # Admins have access to all records
        return patient_record['medical_data']
    elif personnel_role in ['doctor', 'nurse']:
        # Doctors and nurses only have access if they are assigned to the patient
        if personnel_id in patient_record['assigned_staff']:
            return patient_record['medical_data']
        else:
            raise PermissionError("Unauthorized access")
    else:
        # Unknown role - deny access
        raise PermissionError("Unauthorized access")
