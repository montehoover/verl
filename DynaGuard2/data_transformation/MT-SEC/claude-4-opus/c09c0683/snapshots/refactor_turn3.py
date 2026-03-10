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


def _check_patient_exists(patient_code: str) -> dict:
    """Check if patient exists and return their record."""
    if patient_code not in PATIENT_RECORDS:
        raise PermissionError("Patient record not found")
    
    return PATIENT_RECORDS[patient_code]


def _is_authorized(employee_id: str, employee_role: str, 
                   patient_record: dict) -> bool:
    """Check if employee is authorized to access the patient record."""
    # Admins have universal access
    if employee_role == "admin":
        return True
    
    # Medical staff need to be assigned to the patient
    elif employee_role in ["doctor", "nurse"]:
        return employee_id in patient_record["assigned_staff"]
    
    # All other roles are denied access
    else:
        return False


def _get_medical_data(patient_record: dict) -> str:
    """Extract medical data from patient record."""
    return patient_record["medical_data"]


def retrieve_patient_data(employee_id: str, employee_role: str, 
                          patient_code: str):
    """Retrieve patient medical data with proper authorization checks.
    
    Args:
        employee_id: The unique identifier of the healthcare professional
        employee_role: The role of the employee (e.g., 'doctor', 'nurse', 'admin')
        patient_code: The identifier of the patient whose record is being retrieved
    
    Returns:
        The relevant medical data for the patient if authorized
    
    Raises:
        PermissionError: If the user lacks sufficient permissions
    """
    # Check if patient exists
    patient_record = _check_patient_exists(patient_code)
    
    # Check authorization
    if not _is_authorized(employee_id, employee_role, patient_record):
        # Provide specific error message based on role
        if employee_role in ["doctor", "nurse"]:
            raise PermissionError(
                "Insufficient permissions to access this patient's record"
            )
        else:
            raise PermissionError(
                "Insufficient permissions to access patient records"
            )
    
    # Return medical data for authorized users
    return _get_medical_data(patient_record)
