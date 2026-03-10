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

def retrieve_patient_data(employee_id: str, employee_role: str, patient_code: str) -> str:
    """
    Retrieve a patient's medical record in a controlled manner.
    
    Args:
        employee_id: The unique identifier of the healthcare professional
        employee_role: The role of the employee (e.g., 'doctor', 'nurse', 'admin')
        patient_code: The identifier of the patient whose record is being retrieved
    
    Returns:
        The relevant medical data for the patient if authorized
    
    Raises:
        PermissionError: If the employee is not authorized to access the patient's record
    """
    # Check if the patient exists in the records
    if patient_code not in PATIENT_RECORDS:
        raise PermissionError(f"Patient record for {patient_code} not found")
    
    patient_record = PATIENT_RECORDS[patient_code]
    
    # Check if the employee is authorized to access this patient's record
    if employee_id not in patient_record["assigned_staff"]:
        raise PermissionError(f"Employee {employee_id} is not authorized to access patient {patient_code}'s record")
    
    # Return the medical data
    return patient_record["medical_data"]
