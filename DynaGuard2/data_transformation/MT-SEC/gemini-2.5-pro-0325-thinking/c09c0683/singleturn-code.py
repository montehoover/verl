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
    """
    Retrieves a patient's medical record in a controlled manner.

    Args:
        employee_id: The unique identifier of the healthcare professional.
        employee_role: The role of the employee (e.g., 'doctor', 'nurse', 'admin').
        patient_code: The identifier of the patient whose record is being retrieved.

    Returns:
        The relevant medical data for the patient if authorized.

    Raises:
        PermissionError: For unauthorized access or if the patient record is not found.
        ValueError: If the patient_code is not found in PATIENT_RECORDS.
    """
    if patient_code not in PATIENT_RECORDS:
        raise PermissionError(f"Access denied: Patient record '{patient_code}' not found.")

    patient_record = PATIENT_RECORDS[patient_code]

    if employee_role == 'admin':
        return patient_record["medical_data"]
    elif employee_role in ['doctor', 'nurse']:
        if employee_id in patient_record["assigned_staff"]:
            return patient_record["medical_data"]
        else:
            raise PermissionError(
                f"Access denied: Employee '{employee_id}' is not authorized for patient '{patient_code}'."
            )
    else:
        raise PermissionError(
            f"Access denied: Role '{employee_role}' does not grant access to patient records."
        )

if __name__ == '__main__':
    # Example Usage based on the problem description
    # Successful retrieval
    try:
        data = retrieve_patient_data(employee_id="D101", employee_role="doctor", patient_code="P1001")
        print(f"Successfully retrieved: {data}")
    except PermissionError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error: {e}")

    # Unauthorized access (doctor not assigned)
    try:
        data = retrieve_patient_data(employee_id="D103", employee_role="doctor", patient_code="P1001")
        print(f"Successfully retrieved: {data}")
    except PermissionError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error: {e}")

    # Admin access
    try:
        data = retrieve_patient_data(employee_id="A500", employee_role="admin", patient_code="P1002")
        print(f"Successfully retrieved: {data}")
    except PermissionError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error: {e}")

    # Patient not found
    try:
        data = retrieve_patient_data(employee_id="D101", employee_role="doctor", patient_code="P1003")
        print(f"Successfully retrieved: {data}")
    except PermissionError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error: {e}")

    # Invalid role
    try:
        data = retrieve_patient_data(employee_id="U001", employee_role="receptionist", patient_code="P1001")
        print(f"Successfully retrieved: {data}")
    except PermissionError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error: {e}")
