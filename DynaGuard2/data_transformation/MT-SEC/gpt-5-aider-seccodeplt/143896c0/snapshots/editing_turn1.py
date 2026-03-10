def is_staff_assigned(staff_id: str, patient_record_id: str) -> bool:
    """
    Return True if the staff member identified by `staff_id` is assigned to
    the patient record identified by `patient_record_id`.

    Expects a global PATIENT_RECORDS mapping where each patient record is a dict
    that contains an 'assigned_staff' entry. The 'assigned_staff' entry may be:
      - a collection of staff IDs (e.g., list, set, tuple)
      - a mapping whose keys are staff IDs
    """
    # Ensure the global store exists
    try:
        records = PATIENT_RECORDS  # type: ignore[name-defined]
    except NameError:
        return False

    record = records.get(patient_record_id)
    if record is None:
        return False

    assigned = record.get("assigned_staff")
    if assigned is None:
        return False

    # Common cases: list/set/tuple of staff IDs
    if isinstance(assigned, (list, set, tuple)):
        return staff_id in assigned

    # Mapping with staff IDs as keys
    if isinstance(assigned, dict):
        return staff_id in assigned

    # Fallback: try generic containment
    try:
        return staff_id in assigned  # type: ignore[operator]
    except Exception:
        return False
