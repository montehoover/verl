import logging

# Basic logging configuration
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

def _is_authorized(staff_id: str, staff_role: str, patient_record: dict) -> bool:
    """
    Checks if the staff member is authorized to access the patient record.

    Args:
        staff_id: The ID of the healthcare professional.
        staff_role: The role of the healthcare worker.
        patient_record: The patient's record dictionary.

    Returns:
        True if authorized, False otherwise.
    """
    if staff_role == 'admin':
        return True
    if staff_id in patient_record.get("assigned_staff", []):
        return True
    return False

def _get_patient_medical_data(patient_record: dict) -> str:
    """
    Retrieves the medical data from a patient record.

    Args:
        patient_record: The patient's record dictionary.

    Returns:
        The patient's medical data.
    """
    return patient_record["medical_data"]

def retrieve_patient_data(staff_id: str, staff_role: str, patient_id: str):
    """
    Facilitates healthcare staff in retrieving a patient's medical record.

    Args:
        staff_id: The ID of the healthcare professional requesting access.
        staff_role: The role of the healthcare worker (e.g., 'doctor', 'nurse', 'admin').
        patient_id: The identifier of the patient whose record is being accessed.

    Returns:
        The patient's medical data if authorized.

    Raises:
        ValueError: If the patient_id is not found in PATIENT_RECORDS.
        PermissionError: If the access is unauthorized.
    """
    logging.info(f"Access attempt: Staff ID='{staff_id}', Role='{staff_role}', Patient ID='{patient_id}'")

    if patient_id not in PATIENT_RECORDS:
        logging.warning(f"Patient record not found for ID: {patient_id}. Staff ID='{staff_id}', Role='{staff_role}'")
        raise ValueError(f"Patient record not found for ID: {patient_id}")

    patient_record = PATIENT_RECORDS[patient_id]

    if _is_authorized(staff_id, staff_role, patient_record):
        logging.info(f"Access GRANTED: Staff ID='{staff_id}', Role='{staff_role}', Patient ID='{patient_id}'")
        return _get_patient_medical_data(patient_record)
    else:
        logging.warning(f"Access DENIED: Staff ID='{staff_id}', Role='{staff_role}', Patient ID='{patient_id}'. Reason: Unauthorized.")
        raise PermissionError(f"Staff member {staff_id} with role {staff_role} is not authorized to access patient {patient_id}'s data.")
