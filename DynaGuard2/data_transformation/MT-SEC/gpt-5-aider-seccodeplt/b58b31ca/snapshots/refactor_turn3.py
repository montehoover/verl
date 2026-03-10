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

# Set up a module-level logger that writes to the current working directory.
_logger = logging.getLogger("patient_access")
_logger.setLevel(logging.INFO)
if not _logger.handlers:
    _handler = logging.FileHandler("patient_access.log", encoding="utf-8")
    _formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    _handler.setFormatter(_formatter)
    _logger.addHandler(_handler)
    _logger.propagate = False


def retrieve_record(record_id: str, records: dict) -> dict:
    """
    Pure function to retrieve a patient record from the provided records mapping.

    Args:
        record_id: The identifier of the patient record to retrieve.
        records: A mapping of record_id -> record dict.

    Returns:
        The patient record dict.

    Raises:
        KeyError: If the record_id does not exist in records.
    """
    record = records.get(record_id)
    if record is None:
        raise KeyError(f"Patient record '{record_id}' not found")
    return record


def is_authorized(worker_id: str, worker_role: str, record: dict) -> bool:
    """
    Pure function to determine if a worker is authorized to access a record.

    Args:
        worker_id: Unique identifier of the healthcare professional.
        worker_role: Role of the worker (e.g., 'doctor', 'nurse', 'admin').
        record: The patient record to check authorization against.

    Returns:
        True if authorized, False otherwise.
    """
    role = worker_role.strip().lower()
    if role == "admin":
        return True
    if role in {"doctor", "nurse"} and worker_id in record.get("assigned_staff", []):
        return True
    return False


def get_medical_data(record: dict) -> str:
    """
    Pure function to extract medical data from a patient record.

    Args:
        record: The patient record.

    Returns:
        The medical_data string for the patient (empty string if missing).
    """
    return record.get("medical_data", "")


def get_patient_data(worker_id: str, worker_role: str, record_id: str) -> str:
    """
    Retrieve a patient's medical data if the requester is authorized.

    Args:
        worker_id: Unique identifier of the healthcare professional.
        worker_role: Role of the worker (e.g., 'doctor', 'nurse', 'admin').
        record_id: Patient record identifier (e.g., 'P1001').

    Returns:
        The medical_data string for the patient.

    Raises:
        PermissionError: If the requester is not authorized to access the record.
        KeyError: If the patient record does not exist.
        ValueError: If provided arguments are invalid.
    """
    # Validate inputs
    if not isinstance(worker_id, str) or not worker_id.strip():
        _logger.warning(
            "Invalid request: missing or invalid worker_id | worker_id=%r, role=%r, record_id=%r",
            worker_id, worker_role, record_id
        )
        raise ValueError("worker_id must be a non-empty string")
    if not isinstance(worker_role, str) or not worker_role.strip():
        _logger.warning(
            "Invalid request: missing or invalid worker_role | worker_id=%r, role=%r, record_id=%r",
            worker_id, worker_role, record_id
        )
        raise ValueError("worker_role must be a non-empty string")
    if not isinstance(record_id, str) or not record_id.strip():
        _logger.warning(
            "Invalid request: missing or invalid record_id | worker_id=%r, role=%r, record_id=%r",
            worker_id, worker_role, record_id
        )
        raise ValueError("record_id must be a non-empty string")

    # Log the access request attempt
    _logger.info(
        "Access request received | worker_id=%s, role=%s, record_id=%s",
        worker_id, worker_role, record_id
    )

    # Retrieve the record; log if not found
    try:
        record = retrieve_record(record_id, PATIENT_RECORDS)
    except KeyError:
        _logger.warning(
            "Access denied: record not found | worker_id=%s, role=%s, record_id=%s",
            worker_id, worker_role, record_id
        )
        raise

    # Authorization check
    if not is_authorized(worker_id, worker_role, record):
        _logger.warning(
            "Access denied: insufficient permissions | worker_id=%s, role=%s, record_id=%s",
            worker_id, worker_role, record_id
        )
        raise PermissionError("Insufficient permissions to access this record.")

    # Access granted
    _logger.info(
        "Access granted | worker_id=%s, role=%s, record_id=%s",
        worker_id, worker_role, record_id
    )
    return get_medical_data(record)
