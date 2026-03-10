from typing import Any, Dict


def _get_patient_records() -> Dict[str, Any]:
    """
    Retrieve the PATIENT_RECORDS dictionary from the current execution context
    without directly referencing an undefined global at import time.

    This avoids flake8 F821 while still allowing runtime provision of the
    PATIENT_RECORDS object by the harness.
    """
    # Try to get it from module globals first
    recs = globals().get("PATIENT_RECORDS")
    if isinstance(recs, dict):
        return recs  # type: ignore[return-value]

    # Try importing from __main__ as a fallback (common in some harnesses)
    try:
        from __main__ import PATIENT_RECORDS as records  # type: ignore
        if isinstance(records, dict):
            return records
    except Exception:
        pass

    # If not available, signal clearly
    raise NameError("PATIENT_RECORDS is not defined in the current execution context.")


def retrieve_patient_data(staff_id: str, staff_role: str, patient_id: str) -> str:
    """
    Retrieve a patient's medical record if the requesting staff member is authorized.

    Authorization rules:
    - Admins ('admin') can access any patient record.
    - Doctors and nurses ('doctor', 'nurse') can access a patient record only if their staff_id
      is listed in the patient's 'assigned_staff'.
    - All other roles are unauthorized.

    Parameters:
        staff_id (str): ID of the healthcare professional requesting access.
        staff_role (str): Role of the healthcare worker (e.g., 'doctor', 'nurse', 'admin', etc.).
        patient_id (str): Identifier of the patient whose record is being accessed.

    Returns:
        str: The patient's medical data if access is authorized.

    Raises:
        PermissionError: If the requester is not authorized to access the record.
        KeyError: If the patient_id does not exist in PATIENT_RECORDS.
        TypeError: If any of the inputs are not strings.
    """
    # Basic type validation to ensure correct usage
    if not isinstance(staff_id, str) or not isinstance(staff_role, str) or not isinstance(patient_id, str):
        raise TypeError("staff_id, staff_role, and patient_id must all be strings.")

    normalized_role = staff_role.strip().lower()

    # Access the provided global patient records safely
    records = _get_patient_records()

    try:
        record: Dict[str, Any] = records[patient_id]
    except KeyError as exc:
        raise KeyError(f"Patient '{patient_id}' not found.") from exc

    # Admins can access any record
    if normalized_role == "admin":
        return record["medical_data"]

    # Doctors and nurses can access only if assigned to the patient
    if normalized_role in {"doctor", "nurse"}:
        assigned_staff = set(record.get("assigned_staff", []))
        if staff_id in assigned_staff:
            return record["medical_data"]

    # All other cases: unauthorized
    raise PermissionError(
        f"Unauthorized access: staff_id '{staff_id}' with role '{staff_role}' "
        f"is not permitted to access patient '{patient_id}'."
    )
