import logging

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

# Dedicated audit logger for access events.
audit_logger = logging.getLogger("audit.access")
# Ensure library-safe behavior; consuming app can configure handlers/formatters.
if not audit_logger.handlers:
    audit_logger.addHandler(logging.NullHandler())


def fetch_patient_record(patient_ref: str, records: dict) -> dict:
    """
    Pure function to retrieve a patient record from the provided records mapping.

    Args:
        patient_ref: The identifier of the patient whose record is being accessed.
        records: A mapping of patient_id to patient record dictionaries.

    Returns:
        The patient's record dictionary.

    Raises:
        KeyError: If the patient_ref does not exist in records.
    """
    record = records.get(patient_ref)
    if record is None:
        raise KeyError(f"Patient record not found: {patient_ref}")
    return record


def is_authorized(staff_num: str, role: str, record: dict) -> bool:
    """
    Pure function to determine if the requester is authorized to access the given record.

    Rules:
    - 'admin' role can access any record.
    - Otherwise, staff_num must be listed in record['assigned_staff'].

    Args:
        staff_num: The staff ID of the requester.
        role: The role of the requester.
        record: The patient's record.

    Returns:
        True if authorized, False otherwise.
    """
    normalized_role = role.strip().lower()
    if normalized_role == "admin":
        return True

    assigned_staff = set(record.get("assigned_staff", []))
    return staff_num in assigned_staff


def _audit_log(event: str, staff_num: str, role: str, patient_ref: str, *, level: int = logging.INFO, reason: str = "") -> None:
    """
    Internal helper to emit concise, structured audit logs without sensitive data.
    Example message:
        ACCESS_GRANTED staff=D101 role=doctor patient=P1001
        ACCESS_DENIED staff=D999 role=doctor patient=P1001 reason=unauthorized
    """
    msg = f"{event} staff={staff_num} role={role} patient={patient_ref}"
    if reason:
        msg = f"{msg} reason={reason}"
    audit_logger.log(level, msg)


def get_patient_data(staff_num: str, role: str, patient_ref: str) -> str:
    """
    Retrieve a patient's medical data if the requester is authorized.

    Authorization rules:
    - Admin can access any patient's record.
    - Other roles (e.g., doctor, nurse) must be explicitly listed in the patient's assigned_staff.

    Raises:
        KeyError: If the patient_ref does not exist.
        PermissionError: If the requester is not authorized to access the record.

    Returns:
        str: The patient's medical data.
    """
    # Always log the incoming request (without sensitive data).
    _audit_log("ACCESS_REQUEST", str(staff_num), str(role), str(patient_ref), level=logging.INFO)

    if not isinstance(staff_num, str) or not isinstance(role, str) or not isinstance(patient_ref, str):
        _audit_log(
            "ACCESS_DENIED",
            str(staff_num),
            str(role),
            str(patient_ref),
            level=logging.WARNING,
            reason="invalid_arguments",
        )
        raise TypeError("staff_num, role, and patient_ref must all be strings.")

    try:
        record = fetch_patient_record(patient_ref, PATIENT_RECORDS)
    except KeyError:
        _audit_log("ACCESS_DENIED", staff_num, role, patient_ref, level=logging.WARNING, reason="record_not_found")
        raise

    if not is_authorized(staff_num, role, record):
        _audit_log("ACCESS_DENIED", staff_num, role, patient_ref, level=logging.WARNING, reason="unauthorized")
        raise PermissionError(
            f"Unauthorized access: staff {staff_num} with role '{role}' is not permitted to access patient {patient_ref}."
        )

    _audit_log("ACCESS_GRANTED", staff_num, role, patient_ref, level=logging.INFO)
    return record.get("medical_data")
