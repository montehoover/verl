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


def has_access(staff_id: str, role: str, patient_record_id: str) -> bool:
    """
    Determine if a staff member with a given role has access to a patient's record.

    Rules:
      - 'admin' has access to any existing patient record.
      - 'doctor' and 'nurse' must be assigned to the patient to have access.
        If the record stores role information per staff assignment, it must match.
      - All other roles: no access.

    The global PATIENT_RECORDS is expected to be a mapping of patient_record_id to a dict
    that may contain:
      - 'assigned_staff': can be
          * a collection of staff IDs
          * a mapping of staff_id -> role (str), roles (list/set/tuple), or an object
            with 'role' or 'roles' fields.
    """
    # Validate global records presence
    try:
        records = PATIENT_RECORDS  # type: ignore[name-defined]
    except NameError:
        return False

    record = records.get(patient_record_id)
    if record is None:
        return False

    role_norm = (role or "").strip().lower()
    if role_norm == "admin":
        # Admins can access any existing record
        return True

    if role_norm in {"doctor", "nurse"}:
        # Must be assigned
        if not is_staff_assigned(staff_id, patient_record_id):
            return False

        # If assignment includes role details, ensure it matches
        assigned = record.get("assigned_staff")
        if isinstance(assigned, dict) and staff_id in assigned:
            details = assigned[staff_id]

            # Simple case: stored as a single role string
            if isinstance(details, str):
                return details.strip().lower() == role_norm

            # Stored as an iterable of roles
            if isinstance(details, (list, set, tuple)):
                try:
                    roles_norm = {str(r).strip().lower() for r in details}
                except Exception:
                    roles_norm = set()
                return role_norm in roles_norm

            # Stored as a dict with 'role' or 'roles'
            if isinstance(details, dict):
                if "role" in details:
                    r = details.get("role")
                    if isinstance(r, str):
                        return r.strip().lower() == role_norm
                    if isinstance(r, (list, set, tuple)):
                        try:
                            roles_norm = {str(x).strip().lower() for x in r}
                        except Exception:
                            roles_norm = set()
                        return role_norm in roles_norm
                if "roles" in details:
                    r = details.get("roles")
                    if isinstance(r, str):
                        return r.strip().lower() == role_norm
                    if isinstance(r, (list, set, tuple)):
                        try:
                            roles_norm = {str(x).strip().lower() for x in r}
                        except Exception:
                            roles_norm = set()
                        return role_norm in roles_norm

            # Unknown structure: fall back to assignment being sufficient
            return True

        # Assigned without role details: grant access
        return True

    # All other roles are not permitted
    return False
