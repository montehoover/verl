import datetime

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

ACCESS_LOG = []

def retrieve_patient_data(employee_id: str, employee_role: str, patient_code: str):
    access_time = datetime.datetime.now()
    access_granted = False
    
    try:
        if patient_code not in PATIENT_RECORDS:
            raise ValueError(f"Patient {patient_code} not found")
        
        if employee_role not in ROLE_ACCESS:
            raise ValueError(f"Invalid role: {employee_role}")
        
        patient_record = PATIENT_RECORDS[patient_code]
        access_level = ROLE_ACCESS[employee_role]
        
        if access_level == "full":
            access_granted = True
            result = patient_record
        elif access_level == "restricted":
            if employee_id in patient_record["assigned_staff"]:
                access_granted = True
                result = {
                    "patient_id": patient_record["patient_id"],
                    "medical_data": patient_record["medical_data"]
                }
            else:
                raise PermissionError(f"Nurse {employee_id} is not assigned to patient {patient_code}")
        elif access_level == "basic":
            access_granted = True
            result = {
                "patient_id": patient_record["patient_id"]
            }
        else:
            raise PermissionError(f"Insufficient permissions for role: {employee_role}")
        
        ACCESS_LOG.append({
            "timestamp": access_time,
            "employee_id": employee_id,
            "employee_role": employee_role,
            "patient_code": patient_code,
            "access_granted": access_granted
        })
        
        return result
        
    except (ValueError, PermissionError) as e:
        ACCESS_LOG.append({
            "timestamp": access_time,
            "employee_id": employee_id,
            "employee_role": employee_role,
            "patient_code": patient_code,
            "access_granted": access_granted
        })
        raise
