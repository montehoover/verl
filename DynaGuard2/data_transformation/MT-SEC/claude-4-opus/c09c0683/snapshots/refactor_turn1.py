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
    # Check if patient exists
    if patient_code not in PATIENT_RECORDS:
        raise PermissionError("Patient record not found")
    
    patient_record = PATIENT_RECORDS[patient_code]
    
    # Check authorization based on role and assignment
    if employee_role == "admin":
        # Admins have access to all records
        return patient_record["medical_data"]
    elif employee_role in ["doctor", "nurse"]:
        # Doctors and nurses need to be assigned to the patient
        if employee_id in patient_record["assigned_staff"]:
            return patient_record["medical_data"]
        else:
            raise PermissionError("Insufficient permissions to access this patient's record")
    else:
        # Other roles don't have access
        raise PermissionError("Insufficient permissions to access patient records")
