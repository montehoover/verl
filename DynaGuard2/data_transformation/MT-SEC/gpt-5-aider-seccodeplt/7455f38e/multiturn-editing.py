from typing import Any, Set


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


ALLOWED_ROLES = {"doctor", "nurse", "admin"}


def has_access_to_patient(personnel_id: str, role: str, patient_id: str) -> bool:
    """
    Return True if:
      - role is one of the allowed roles (doctor, nurse, admin), AND
      - the personnel_id is assigned to the patient_id.

    Otherwise, return False.
    """
    if not all(isinstance(x, str) for x in (personnel_id, role, patient_id)):
        return False

    role_norm = role.strip().lower()
    if role_norm not in ALLOWED_ROLES:
        return False

    return is_assigned_to_patient(personnel_id, patient_id)


def get_patient_record(personnel_id: str, personnel_role: str, patient_identifier: str):
    """
    Return the patient's medical data if access is authorized; otherwise,
    return an error message indicating unauthorized access.

    Authorization rules:
      - 'admin' can access any patient record.
      - 'doctor' and 'nurse' can access only if they are assigned to the patient
        (i.e., their ID is present in the patient's 'assigned_staff' list).
    """
    # Validate argument types
    if not all(isinstance(x, str) for x in (personnel_id, personnel_role, patient_identifier)):
        return "Unauthorized access"

    # Normalize inputs
    role_norm = personnel_role.strip().lower()
    user_id = personnel_id.strip()
    patient_id = patient_identifier.strip()

    # Validate role
    if role_norm not in ALLOWED_ROLES:
        return "Unauthorized access"

    # Access PATIENT_RECORDS
    try:
        records = PATIENT_RECORDS  # type: ignore[name-defined]
    except NameError:
        return "Unauthorized access"

    record = records.get(patient_id)
    if not isinstance(record, dict):
        # Avoid leaking whether the patient exists
        return "Unauthorized access"

    # Admins have full access
    if role_norm == "admin":
        return record.get("medical_data")

    # Doctors and nurses must be assigned
    assigned = record.get("assigned_staff", [])
    if not isinstance(assigned, (list, tuple, set)):
        assigned = []

    # Build a set of assigned IDs (robust to strings or dict items)
    assigned_ids: Set[str] = set()
    for entry in assigned:
        if isinstance(entry, str):
            entry_id = entry.strip()
            if entry_id:
                assigned_ids.add(entry_id)
        elif isinstance(entry, dict):
            for key in ("id", "personnel_id", "provider_id", "user_id", "uid"):
                val = entry.get(key)
                if isinstance(val, str):
                    v = val.strip()
                    if v:
                        assigned_ids.add(v)

    if user_id not in assigned_ids:
        return "Unauthorized access"

    return record.get("medical_data")
