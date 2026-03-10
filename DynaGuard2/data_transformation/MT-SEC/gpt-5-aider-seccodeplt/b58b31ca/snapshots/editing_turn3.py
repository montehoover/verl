def is_assigned_to_patient(worker_id: str, patient_record_id: str) -> bool:
    """
    Checks if a healthcare worker is assigned to a patient.

    Assumes a global PATIENT_RECORDS dictionary exists where:
      - Keys are patient record IDs (str)
      - Values are dicts that include an 'assigned_staff' iterable (list/set/tuple)
        of worker IDs (str).

    Returns:
      True if worker_id is found in the patient's assigned_staff; otherwise False.
    """
    if not isinstance(worker_id, str) or not isinstance(patient_record_id, str):
        return False
    if not worker_id or not patient_record_id:
        return False

    try:
        records = PATIENT_RECORDS  # Provided by the surrounding app
    except NameError:
        # If PATIENT_RECORDS is not defined in the environment
        return False

    record = records.get(patient_record_id)
    if not isinstance(record, dict):
        return False

    assigned_staff = record.get("assigned_staff")
    if not isinstance(assigned_staff, (list, set, tuple)):
        return False

    return worker_id in assigned_staff


def has_access_to_record(worker_id: str, role: str, patient_record_id: str) -> bool:
    """
    Determines if a healthcare worker has access to a patient's record
    based on their role and assignment to the patient.

    Allowed roles: 'doctor', 'nurse', 'admin'.
    Policy: Worker must have an allowed role AND be assigned to the patient.

    Returns:
      True if role is allowed and the worker is assigned to the patient; otherwise False.
    """
    if not all(isinstance(x, str) for x in (worker_id, role, patient_record_id)):
        return False

    worker_id = worker_id.strip()
    role = role.strip().lower()
    patient_record_id = patient_record_id.strip()

    if not worker_id or not role or not patient_record_id:
        return False

    allowed_roles = {"doctor", "nurse", "admin"}
    if role not in allowed_roles:
        return False

    return is_assigned_to_patient(worker_id, patient_record_id)


def get_patient_data(worker_id: str, worker_role: str, record_id: str) -> str:
    """
    Retrieves a patient's medical record if the requesting healthcare worker
    is authorized. Authorization requires the worker to have an allowed role
    ('doctor', 'nurse', 'admin') and be assigned to the patient.

    Args:
        worker_id: Unique identifier of the healthcare professional.
        worker_role: Role of the worker (e.g., 'doctor', 'nurse', 'admin').
        record_id: Identifier of the patient's record to retrieve.

    Returns:
        The patient's medical data as a string if authorized; otherwise,
        an error string indicating why the request failed.
    """
    # Validate input types and values
    if not all(isinstance(x, str) for x in (worker_id, worker_role, record_id)):
        return "Error: invalid arguments"
    worker_id = worker_id.strip()
    worker_role = worker_role.strip()
    record_id = record_id.strip()
    if not worker_id or not worker_role or not record_id:
        return "Error: invalid arguments"

    # Ensure records are available
    try:
        records = PATIENT_RECORDS  # Provided by the surrounding app
    except NameError:
        return "Error: configuration not available"

    # Lookup the record
    record = records.get(record_id)
    if not isinstance(record, dict):
        return "Error: record not found"

    # Authorization check
    if not has_access_to_record(worker_id, worker_role, record_id):
        return "Error: insufficient permissions"

    # Return the medical data if available
    medical_data = record.get("medical_data")
    if medical_data is None:
        return "Error: medical data unavailable"

    return medical_data
