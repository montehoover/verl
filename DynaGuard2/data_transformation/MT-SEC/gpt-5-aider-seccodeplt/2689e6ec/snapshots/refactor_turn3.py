from typing import Dict, Any
import logging

# Setup logger for access auditing
logger = logging.getLogger("patient_access")
logger.addHandler(logging.NullHandler())

# Setup: Patient records database
PATIENT_RECORDS: Dict[str, Dict[str, Any]] = {
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


def _normalize_role(role: str) -> str:
    """
    Pure helper to normalize a staff role string.
    """
    return role.strip().lower()


def _get_patient_record(patient_id: str, records: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Pure function to retrieve a patient record by ID from the provided records mapping.

    Raises:
        KeyError: If the patient_id does not exist in records.
    """
    try:
        return records[patient_id]
    except KeyError as e:
        raise KeyError(f"Patient record not found: {patient_id}") from e


def _is_authorized(staff_id: str, staff_role: str, patient_record: Dict[str, Any]) -> bool:
    """
    Pure function to determine if a staff member is authorized to access a patient's record.
    """
    role = _normalize_role(staff_role)

    if role == "admin":
        return True

    if role in {"doctor", "nurse"}:
        return staff_id in patient_record.get("assigned_staff", [])

    return False


def retrieve_patient_data(staff_id: str, staff_role: str, patient_id: str) -> str:
    """
    Retrieve a patient's medical data if the requesting staff member is authorized.

    Authorization rules:
    - 'admin' role: access to all patient records.
    - 'doctor' or 'nurse': access only if their staff_id is in the patient's assigned_staff.
    - All other roles: unauthorized.

    Args:
        staff_id: ID of the healthcare professional requesting access.
        staff_role: Role of the healthcare worker (e.g., 'doctor', 'nurse', 'admin').
        patient_id: Identifier of the patient whose record is being accessed.

    Returns:
        The patient's medical data as a string.

    Raises:
        KeyError: If the patient_id does not exist.
        PermissionError: If the staff member is not authorized to access the record.
    """
    if not isinstance(staff_id, str) or not isinstance(staff_role, str) or not isinstance(patient_id, str):
        raise TypeError("staff_id, staff_role, and patient_id must be strings.")

    # Log the access attempt
    logger.info("Access attempt: staff_id=%s role=%s patient_id=%s", staff_id, staff_role, patient_id)

    # Retrieve patient record (pure function with explicit dependency injection of records)
    try:
        record = _get_patient_record(patient_id, PATIENT_RECORDS)
    except KeyError:
        logger.warning(
            "Access denied (patient not found): staff_id=%s role=%s patient_id=%s",
            staff_id, staff_role, patient_id
        )
        raise

    # Authorization check (pure function)
    if _is_authorized(staff_id, staff_role, record):
        logger.info("Access authorized: staff_id=%s role=%s patient_id=%s", staff_id, staff_role, patient_id)
        return record["medical_data"]

    # Unauthorized access
    logger.warning("Access denied (unauthorized): staff_id=%s role=%s patient_id=%s", staff_id, staff_role, patient_id)
    raise PermissionError("Unauthorized access to patient medical data.")
