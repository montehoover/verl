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

def _get_patient_record_from_db(patient_record_id: str, db: dict) -> dict:
    """
    Retrieves a patient record from the database.

    Args:
        patient_record_id: The ID of the patient record to retrieve.
        db: The database dictionary containing patient records.

    Returns:
        The patient record.

    Raises:
        PermissionError: If the record is not found.
    """
    if patient_record_id not in db:
        # Mimic existing behavior: raise PermissionError to avoid leaking info
        # about record existence vs. access denial.
        raise PermissionError(f"Record {patient_record_id} not found or access denied.")
    return db[patient_record_id]

def _is_user_authorized_for_record(staff_id: str, staff_role: str, record: dict) -> bool:
    """
    Checks if a staff member is authorized to access a given patient record.

    Args:
        staff_id: The unique identifier of the healthcare professional.
        staff_role: The role of the staff member.
        record: The patient record dictionary.

    Returns:
        True if authorized, False otherwise.
    """
    if staff_role == 'admin':
        return True
    if staff_id in record.get("assigned_staff", []):
        return True
    return False

def _get_medical_data_from_record(record: dict) -> str:
    """
    Extracts medical data from a patient record.

    Args:
        record: The patient record dictionary.

    Returns:
        The medical data string.
    """
    return record["medical_data"]

def view_patient_record(staff_id: str, staff_role: str, patient_record_id: str):
    """
    Retrieves a patient's medical record in a controlled manner.

    Args:
        staff_id: The unique identifier of the healthcare professional.
        staff_role: The role of the staff member (e.g., 'doctor', 'nurse', 'admin').
        patient_record_id: The identifier of the patient whose record is being retrieved.

    Returns:
        The relevant medical data for the patient if authorized.

    Raises:
        PermissionError: If access is unauthorized or the record does not exist.
    """
    # Step 1: Get the record (handles "not found" by raising PermissionError)
    record = _get_patient_record_from_db(patient_record_id, PATIENT_RECORDS)

    # Step 2: Check authorization
    if not _is_user_authorized_for_record(staff_id, staff_role, record):
        raise PermissionError(f"Staff {staff_id} with role {staff_role} is not authorized to view record {patient_record_id}.")

    # Step 3: Retrieve and return medical data
    return _get_medical_data_from_record(record)
