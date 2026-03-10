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


def has_access_to_patient(personnel_id: str, personnel_role: str, patient_record: dict) -> bool:
    """Check if personnel has access to a patient's record based on their role and assignment."""
    if personnel_role == 'admin':
        return True
    elif personnel_role in ['doctor', 'nurse']:
        return personnel_id in patient_record['assigned_staff']
    else:
        return False


def retrieve_patient_data(patient_identifier: str, records: dict) -> dict:
    """Retrieve patient record from the records dictionary."""
    if patient_identifier not in records:
        return None
    return records[patient_identifier]


def get_patient_record(personnel_id: str, personnel_role: str, patient_identifier: str):
    # Retrieve patient record
    patient_record = retrieve_patient_data(patient_identifier, PATIENT_RECORDS)
    
    if patient_record is None:
        raise PermissionError("Unauthorized access")
    
    # Check access control
    if has_access_to_patient(personnel_id, personnel_role, patient_record):
        return patient_record['medical_data']
    else:
        raise PermissionError("Unauthorized access")
