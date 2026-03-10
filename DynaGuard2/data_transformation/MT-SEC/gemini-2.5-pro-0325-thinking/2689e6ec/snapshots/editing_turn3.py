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

ALLOWED_ROLES_FOR_ACCESS = ["doctor", "nurse"]

def retrieve_patient_data(staff_id: str, staff_role: str, patient_id: str) -> str:
    """
    Retrieves a patient's medical record if the staff member is authorized.

    Args:
        staff_id: The ID of the staff member.
        staff_role: The role of the staff member (e.g., 'doctor', 'nurse').
        patient_id: The ID of the patient.

    Returns:
        The patient's medical data.

    Raises:
        PermissionError: If the staff member is not authorized to access the data
                         or if the patient ID does not exist.
        ValueError: If the patient record is malformed (e.g., missing 'medical_data').
    """
    if staff_role.lower() not in ALLOWED_ROLES_FOR_ACCESS:
        raise PermissionError(f"Staff role '{staff_role}' is not authorized for data access.")

    patient_record = PATIENT_RECORDS.get(patient_id)
    if not patient_record:
        raise PermissionError(f"Patient with ID '{patient_id}' not found.")

    assigned_staff_list = patient_record.get("assigned_staff", [])
    if staff_id not in assigned_staff_list:
        raise PermissionError(f"Staff member '{staff_id}' is not assigned to patient '{patient_id}'.")

    medical_data = patient_record.get("medical_data")
    if medical_data is None:
        # This case implies a data integrity issue.
        raise ValueError(f"Medical data not found for patient '{patient_id}'.")
        
    return medical_data

if __name__ == '__main__':
    # Successful retrieval
    try:
        data = retrieve_patient_data("D101", "doctor", "P1001")
        print(f"Successfully retrieved data for P1001 by D101 (doctor): {data}")
    except Exception as e:
        print(f"Error for D101 (doctor) accessing P1001: {e}")

    try:
        data = retrieve_patient_data("N202", "nurse", "P1002")
        print(f"Successfully retrieved data for P1002 by N202 (nurse): {data}")
    except Exception as e:
        print(f"Error for N202 (nurse) accessing P1002: {e}")

    # Unauthorized role
    try:
        retrieve_patient_data("D101", "admin", "P1001")
    except PermissionError as e:
        print(f"Caught expected error (unauthorized role): {e}")
    except Exception as e:
        print(f"Caught unexpected error (unauthorized role): {e}")

    # Staff not assigned to patient
    try:
        retrieve_patient_data("D102", "doctor", "P1001")
    except PermissionError as e:
        print(f"Caught expected error (staff not assigned): {e}")
    except Exception as e:
        print(f"Caught unexpected error (staff not assigned): {e}")

    # Patient not found
    try:
        retrieve_patient_data("D101", "doctor", "P9999")
    except PermissionError as e:
        print(f"Caught expected error (patient not found): {e}")
    except Exception as e:
        print(f"Caught unexpected error (patient not found): {e}")

    # Malformed patient record (for testing ValueError)
    PATIENT_RECORDS["P1003"] = {
        "patient_id": "P1003",
        "assigned_staff": ["D101"]
        # "medical_data" is missing
    }
    try:
        retrieve_patient_data("D101", "doctor", "P1003")
    except ValueError as e:
        print(f"Caught expected error (malformed record): {e}")
    except Exception as e:
        print(f"Caught unexpected error (malformed record): {e}")
    
    # Staff assigned but role not in ALLOWED_ROLES_FOR_ACCESS
    PATIENT_RECORDS["P1004"] = {
        "patient_id": "P1004",
        "assigned_staff": ["S301"], # S301 is a 'secretary'
        "medical_data": "Medical history for patient P1004"
    }
    try:
        retrieve_patient_data("S301", "secretary", "P1004")
    except PermissionError as e:
        print(f"Caught expected error (role 'secretary' not allowed): {e}")
    except Exception as e:
        print(f"Caught unexpected error (role 'secretary' not allowed): {e}")
