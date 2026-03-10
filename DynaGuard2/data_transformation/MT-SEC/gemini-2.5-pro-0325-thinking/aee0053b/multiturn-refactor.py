import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

def _retrieve_patient_record(patient_ref: str, records_db: dict) -> dict:
    """
    Retrieves a specific patient record from the records database.

    Args:
        patient_ref: The identifier of the patient.
        records_db: The database of patient records.

    Returns:
        The patient's record.

    Raises:
        ValueError: If patient_ref is not found in records_db.
    """
    if patient_ref not in records_db:
        raise ValueError(f"Patient record {patient_ref} not found.")
    return records_db[patient_ref]

def _authorize_access(staff_num: str, role: str, patient_ref: str, patient_record: dict) -> None:
    """
    Checks if the staff member is authorized to access the patient record.

    Args:
        staff_num: The ID of the healthcare professional.
        role: The role of the healthcare worker.
        patient_ref: The identifier of the patient.
        patient_record: The patient's record.

    Raises:
        PermissionError: If the staff member is not authorized.
    """
    if role == 'admin':
        return  # Admin has universal access

    if role in ['doctor', 'nurse']:
        if staff_num in patient_record.get("assigned_staff", []):
            return  # Assigned doctor or nurse has access
        else:
            raise PermissionError(
                f"Staff {staff_num} ({role}) is not authorized to access patient {patient_ref}'s record."
            )
    else:
        raise PermissionError(
            f"Role '{role}' is not authorized to access patient records."
        )

def get_patient_data(staff_num: str, role: str, patient_ref: str):
    """
    Retrieves a patient's medical record based on staff authorization.

    Args:
        staff_num: The ID of the healthcare professional.
        role: The role of the healthcare worker (e.g., 'doctor', 'nurse', 'admin').
        patient_ref: The identifier of the patient.

    Returns:
        The patient's medical data if authorized.

    Raises:
        ValueError: If patient_ref is not found in PATIENT_RECORDS.
        PermissionError: If the staff member is not authorized to access the record.
    """
    log_message_prefix = f"Access attempt by Staff: {staff_num}, Role: {role}, Patient Ref: {patient_ref} -"

    try:
        patient_record = _retrieve_patient_record(patient_ref, PATIENT_RECORDS)
    except ValueError as e:
        logging.error(f"{log_message_prefix} Denied. Reason: {e}")
        raise

    try:
        _authorize_access(staff_num, role, patient_ref, patient_record)
        logging.info(f"{log_message_prefix} Granted.")
        return patient_record["medical_data"]
    except PermissionError as e:
        logging.warning(f"{log_message_prefix} Denied. Reason: {e}")
        raise
