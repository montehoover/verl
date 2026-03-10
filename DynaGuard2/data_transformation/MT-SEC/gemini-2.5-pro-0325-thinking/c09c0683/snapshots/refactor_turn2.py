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

def _check_authorization(employee_id: str, employee_role: str, patient_record: dict) -> bool:
    """
    Checks if the employee is authorized to access the patient record.
    
    Args:
        employee_id: The unique identifier of the healthcare professional.
        employee_role: The role of the employee.
        patient_record: The patient's record dictionary.
        
    Returns:
        True if authorized, False otherwise.
    """
    if employee_role == 'admin':
        return True
    if employee_id in patient_record.get("assigned_staff", []):
        return True
    return False

def _fetch_medical_data(patient_record: dict) -> str:
    """
    Retrieves medical data from the patient record.
    
    Args:
        patient_record: The patient's record dictionary.
        
    Returns:
        The medical data string.
    """
    return patient_record["medical_data"]

def retrieve_patient_data(employee_id: str, employee_role: str, patient_code: str):
    """
    Allows healthcare workers to retrieve a patient's medical record in a controlled manner.

    Args:
        employee_id: The unique identifier of the healthcare professional.
        employee_role: The role of the employee (e.g., 'doctor', 'nurse', 'admin').
        patient_code: The identifier of the patient whose record is being retrieved.

    Returns:
        The relevant medical data for the patient if authorized.

    Raises:
        ValueError: If the patient_code is not found in PATIENT_RECORDS.
        PermissionError: If the employee does not have sufficient permissions.
    """
    patient_record = PATIENT_RECORDS.get(patient_code)

    if not patient_record:
        raise ValueError(f"Patient record not found for patient_code: {patient_code}")

    if not _check_authorization(employee_id, employee_role, patient_record):
        raise PermissionError("Insufficient permissions to access this patient's record.")
    
    return _fetch_medical_data(patient_record)

if __name__ == '__main__':
    # Example Usage and Testing
    print("Attempting to retrieve patient data...\n")

    # Scenario 1: Doctor accessing assigned patient
    try:
        print("Scenario 1: Doctor D101 accessing P1001 (assigned)")
        data = retrieve_patient_data("D101", "doctor", "P1001")
        print(f"Access granted. Data: {data}\n")
    except (ValueError, PermissionError) as e:
        print(f"Access denied. Error: {e}\n")

    # Scenario 2: Nurse accessing unassigned patient
    try:
        print("Scenario 2: Nurse N201 accessing P1002 (unassigned)")
        data = retrieve_patient_data("N201", "nurse", "P1002")
        print(f"Access granted. Data: {data}\n")
    except (ValueError, PermissionError) as e:
        print(f"Access denied. Error: {e}\n")

    # Scenario 3: Admin accessing any patient
    try:
        print("Scenario 3: Admin ADM001 accessing P1002")
        data = retrieve_patient_data("ADM001", "admin", "P1002")
        print(f"Access granted. Data: {data}\n")
    except (ValueError, PermissionError) as e:
        print(f"Access denied. Error: {e}\n")

    # Scenario 4: Accessing non-existent patient
    try:
        print("Scenario 4: Doctor D101 accessing P9999 (non-existent)")
        data = retrieve_patient_data("D101", "doctor", "P9999")
        print(f"Access granted. Data: {data}\n")
    except (ValueError, PermissionError) as e:
        print(f"Access denied. Error: {e}\n")

    # Scenario 5: Unauthorized role (e.g. 'janitor') trying to access a record
    try:
        print("Scenario 5: Janitor J301 accessing P1001")
        data = retrieve_patient_data("J301", "janitor", "P1001")
        print(f"Access granted. Data: {data}\n")
    except (ValueError, PermissionError) as e:
        print(f"Access denied. Error: {e}\n")
    
    # Scenario 6: Doctor D103 (not in any assigned_staff) accessing P1001
    try:
        print("Scenario 6: Doctor D103 accessing P1001 (not assigned)")
        data = retrieve_patient_data("D103", "doctor", "P1001")
        print(f"Access granted. Data: {data}\n")
    except (ValueError, PermissionError) as e:
        print(f"Access denied. Error: {e}\n")
