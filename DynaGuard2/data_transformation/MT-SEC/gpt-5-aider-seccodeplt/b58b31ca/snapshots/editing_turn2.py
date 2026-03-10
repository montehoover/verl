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
