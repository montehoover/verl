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

def retrieve_patient_data(employee_id: str, employee_role: str, patient_code: str):
    if patient_code not in PATIENT_RECORDS:
        raise ValueError(f"Patient {patient_code} not found")
    
    patient_record = PATIENT_RECORDS[patient_code]
    
    if employee_id in patient_record["assigned_staff"]:
        return patient_record["medical_data"]
    else:
        raise PermissionError(f"Employee {employee_id} is not authorized to access patient {patient_code}")
