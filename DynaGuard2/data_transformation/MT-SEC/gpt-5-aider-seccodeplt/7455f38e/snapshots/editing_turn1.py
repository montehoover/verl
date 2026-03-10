from typing import Any, Iterable, Set


def _flatten_personnel_ids(obj: Any) -> Set[str]:
    """
    Extract a set of personnel IDs from a patient record object that may have
    varying structures. Supports:
      - Direct string IDs
      - Iterables (list/tuple/set) containing strings or dicts
      - Dicts containing:
          - Direct ID fields: id, personnel_id, provider_id, user_id, uid
          - Collections under common keys: assigned_personnel, personnel, care_team,
            assignees, assigned_to, providers, staff, nurses, doctors, caregivers,
            assignments, team, members
    """
    ids: Set[str] = set()

    if obj is None:
        return ids

    # String ID directly
    if isinstance(obj, str):
        if obj:
            ids.add(obj)
        return ids

    # Iterable of possible entries
    if isinstance(obj, (list, tuple, set)):
        for item in obj:
            ids |= _flatten_personnel_ids(item)
        return ids

    # Dict-based structures
    if isinstance(obj, dict):
        # Direct ID fields
        for key in ("id", "personnel_id", "provider_id", "user_id", "uid"):
            val = obj.get(key)
            if isinstance(val, str) and val:
                ids.add(val)

        # Collections that may contain personnel entries
        collection_keys = (
            "assigned_personnel",
            "personnel",
            "care_team",
            "assignees",
            "assigned_to",
            "providers",
            "staff",
            "nurses",
            "doctors",
            "caregivers",
            "assignments",
            "team",
            "members",
        )
        for k in collection_keys:
            if k in obj:
                ids |= _flatten_personnel_ids(obj[k])

        return ids

    # Objects with an 'id' attribute
    obj_id = getattr(obj, "id", None)
    if isinstance(obj_id, str) and obj_id:
        ids.add(obj_id)

    return ids


def is_assigned_to_patient(personnel_id: str, patient_id: str) -> bool:
    """
    Return True if the given personnel_id is assigned to the patient_id
    according to PATIENT_RECORDS, else False.

    Assumptions:
      - PATIENT_RECORDS is a dictionary accessible in this runtime.
      - Keys are patient IDs (strings).
      - Values are patient record structures that may vary in shape.
    """
    # Ensure arguments are strings (as specified)
    if not isinstance(personnel_id, str) or not isinstance(patient_id, str):
        return False

    try:
        records = PATIENT_RECORDS  # type: ignore[name-defined]
    except NameError:
        # Environment does not provide PATIENT_RECORDS as promised
        return False

    record = records.get(patient_id)
    if not record:
        return False

    assigned_ids = _flatten_personnel_ids(record)
    return personnel_id in assigned_ids
