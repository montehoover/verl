from typing import Any, Dict, Iterable


__all__ = ["is_staff_assigned", "has_access"]


def is_staff_assigned(staff_id: str, patient_id: str) -> bool:
    """
    Check if a staff member is assigned to a specific patient.

    Args:
        staff_id: The staff member's ID.
        patient_id: The patient's ID.

    Returns:
        True if the staff member is listed in the patient's assigned staff; otherwise False.

    Notes:
        Expects a global PATIENT_RECORDS dictionary structured such that:
            PATIENT_RECORDS[patient_id] -> dict with key 'assigned_staff'
        'assigned_staff' may be a list, set, tuple, dict (staff IDs as keys), or a single value.
    """
    # Obtain the patient records from the global scope, defaulting to an empty dict if not present.
    records: Dict[str, Dict[str, Any]] = globals().get("PATIENT_RECORDS", {})  # type: ignore[assignment]

    if not isinstance(staff_id, str) or not isinstance(patient_id, str):
        return False

    patient_record = records.get(patient_id)
    if not isinstance(patient_record, dict):
        return False

    assigned = patient_record.get("assigned_staff")
    if assigned is None:
        return False

    # Normalize assigned staff into an iterable of IDs to check membership.
    if isinstance(assigned, dict):
        collection: Iterable[Any] = assigned.keys()
    elif isinstance(assigned, (list, set, tuple)):
        collection = assigned
    else:
        collection = [assigned]

    return staff_id in collection


def has_access(staff_id: str, role: str, patient_id: str) -> bool:
    """
    Determine whether a staff member has access to a patient's record based on role.

    Rules:
    - 'admin' has access to all records.
    - 'doctor' and 'nurse' have access only if they are assigned to the patient.
    - If the patient's assigned_staff is a dict mapping staff IDs to role info, the provided role
      must match one of the roles associated with that staff for the patient.

    Args:
        staff_id: The staff member's ID.
        role: The staff member's role (e.g., 'doctor', 'nurse', 'admin').
        patient_id: The patient's ID.

    Returns:
        True if the staff member has the appropriate role to access the patient's record; otherwise False.

    Notes:
        Expects a global PATIENT_RECORDS dictionary structured such that:
            PATIENT_RECORDS[patient_id] -> dict with key 'assigned_staff'
        'assigned_staff' may be:
          - a collection (list/set/tuple) of staff IDs,
          - a dict where keys are staff IDs and values may be:
              * a role string (e.g., "doctor"),
              * a collection of role strings (e.g., ["doctor", "nurse"]),
              * a dict either with a 'role' key or role-name keys with truthy values.
    """
    # Basic type checks
    if not isinstance(staff_id, str) or not isinstance(patient_id, str) or not isinstance(role, str):
        return False

    role_norm = role.strip().lower()
    valid_roles = {"doctor", "nurse", "admin"}
    if role_norm not in valid_roles:
        return False

    # Admins have universal access.
    if role_norm == "admin":
        return True

    # For doctor and nurse, require that the staff is assigned to the patient.
    if not is_staff_assigned(staff_id, patient_id):
        return False

    # If assignment carries role-specific information, ensure the role matches.
    records: Dict[str, Dict[str, Any]] = globals().get("PATIENT_RECORDS", {})  # type: ignore[assignment]
    patient_record = records.get(patient_id)
    if not isinstance(patient_record, dict):
        return False

    assigned = patient_record.get("assigned_staff")

    # If assigned is a dict, attempt to derive roles for this staff member.
    if isinstance(assigned, dict):
        assigned_value = assigned.get(staff_id)

        roles_for_staff = set()
        if isinstance(assigned_value, str):
            roles_for_staff.add(assigned_value.lower())
        elif isinstance(assigned_value, (list, set, tuple)):
            roles_for_staff = {str(r).lower() for r in assigned_value}
        elif isinstance(assigned_value, dict):
            if "role" in assigned_value and isinstance(assigned_value["role"], str):
                roles_for_staff.add(assigned_value["role"].lower())
            else:
                for k, v in assigned_value.items():
                    try:
                        if bool(v):
                            roles_for_staff.add(str(k).lower())
                    except Exception:
                        # Non-bool-castable values are ignored
                        pass

        # If roles are specified for the staff and don't include the requested role, deny.
        if roles_for_staff and role_norm not in roles_for_staff:
            return False

    # If there is no role-specific info or it matches, allow access.
    return True
