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

ROLE_ACCESS = {
    "doctor": "full",
    "nurse": "restricted",
    "admin": "basic"
}

def retrieve_patient_data(employee_id: str, employee_role: str, patient_code: str):
    if patient_code not in PATIENT_RECORDS:
        raise ValueError(f"Patient {patient_code} not found")
    
    if employee_role not in ROLE_ACCESS:
        raise ValueError(f"Invalid role: {employee_role}")
    
    patient_record = PATIENT_RECORDS[patient_code]
    access_level = ROLE_ACCESS[employee_role]
    
    if access_level == "full":
        return patient_record
    elif access_level == "restricted":
        if employee_id in patient_record["assigned_staff"]:
            return {
                "patient_id": patient_record["patient_id"],
                "medical_data": patient_record["medical_data"]
            }
        else:
            raise PermissionError(f"Nurse {employee_id} is not assigned to patient {patient_code}")
    elif access_level == "basic":
        return {
            "patient_id": patient_record["patient_id"]
        }
    else:
        raise PermissionError(f"Insufficient permissions for role: {employee_role}")
