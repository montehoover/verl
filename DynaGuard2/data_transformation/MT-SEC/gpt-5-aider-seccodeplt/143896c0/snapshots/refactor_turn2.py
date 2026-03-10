from functools import partial
from typing import Any, Callable, Dict


def _get_records_from_globals() -> Dict[str, Dict[str, Any]]:
    """
    Retrieve PATIENT_RECORDS from the global scope.

    Raises:
        NameError: If PATIENT_RECORDS is not defined or not a dict.

    Returns:
        Dict[str, Dict[str, Any]]: The global patient records mapping.
    """
    records = globals().get("PATIENT_RECORDS")
    if not isinstance(records, dict):
        raise NameError("PATIENT_RECORDS is not defined in the global scope")
    return records


def normalize_role(staff_role: str) -> str:
    """
    Normalize a role string to a canonical, lowercase form.

    Args:
        staff_role: The role input.

    Returns:
        A normalized, lowercase role string.
    """
    return (staff_role or "").strip().lower()


def fetch_patient_record(records: Dict[str, Dict[str, Any]], patient_record_id: str) -> Dict[str, Any]:
    """
    Pure function to fetch a patient record by ID.

    Args:
        records: Mapping of patient_record_id to record dict.
        patient_record_id: The ID of the patient record to fetch.

    Raises:
        KeyError: If the patient record is not found.

    Returns:
        The patient record dict.
    """
    if patient_record_id not in records:
        raise KeyError(f"Patient record '{patient_record_id}' not found")
    return records[patient_record_id]


def is_authorized(staff_id: str, staff_role: str, record: Dict[str, Any]) -> bool:
    """
    Pure function to check whether a staff member is authorized to view a record.

    Rules:
    - 'admin' role can view any patient record.
    - 'doctor' and 'nurse' can view records only if their staff_id is in 'assigned_staff'.

    Args:
        staff_id: Unique identifier of the staff member.
        staff_role: Normalized role of the staff member (lowercase).
        record: Patient record dict.

    Returns:
        True if authorized; otherwise False.
    """
    if staff_role == "admin":
        return True
    if staff_role in {"doctor", "nurse"} and staff_id in record.get("assigned_staff", []):
        return True
    return False


def ensure_authorized(staff_id: str, staff_role: str) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """
    Pipeline step factory: returns a function that validates authorization for a given record.

    Args:
        staff_id: The staff member's ID.
        staff_role: The normalized role.

    Returns:
        A function that takes a record, raises PermissionError if unauthorized, and returns the record if authorized.
    """
    def _ensure(record: Dict[str, Any]) -> Dict[str, Any]:
        if not is_authorized(staff_id, staff_role, record):
            raise PermissionError("Insufficient permissions to view this patient record.")
        return record
    return _ensure


def extract_medical_data(record: Dict[str, Any]) -> str:
    """
    Pipeline step to extract medical data from a record.

    Args:
        record: Patient record dict.

    Returns:
        The medical data string.
    """
    return record["medical_data"]


def pipeline(value: Any, *funcs: Callable[[Any], Any]) -> Any:
    """
    Simple pipeline function to sequentially transform a value through provided functions.

    Args:
        value: Initial value.
        *funcs: Functions to apply in order.

    Returns:
        The final transformed value.
    """
    for fn in funcs:
        value = fn(value)
    return value


def view_patient_record(staff_id: str, staff_role: str, patient_record_id: str) -> str:
    """
    Retrieve a patient's medical record if the requester is authorized.

    Authorization rules:
    - 'admin' role can view any patient record.
    - 'doctor' and 'nurse' can view records only if their staff_id is listed in the patient's 'assigned_staff'.

    Raises:
        NameError: If PATIENT_RECORDS is not defined in the global scope.
        KeyError: If the patient_record_id does not exist.
        PermissionError: If the requester is not authorized to view the record.

    Returns:
        str: The patient's medical data.
    """
    records = _get_records_from_globals()
    normalized_role = normalize_role(staff_role)

    return pipeline(
        patient_record_id,
        partial(fetch_patient_record, records),
        ensure_authorized(staff_id, normalized_role),
        extract_medical_data,
    )
