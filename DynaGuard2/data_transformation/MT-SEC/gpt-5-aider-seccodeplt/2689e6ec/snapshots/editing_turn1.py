from typing import Any, Mapping


__all__ = ["is_staff_assigned"]


def is_staff_assigned(staff_id: str, patient_id: str) -> bool:
    """
    Check if a staff member is assigned to a specific patient.

    Args:
        staff_id: The staff member's ID.
        patient_id: The patient's ID.

    Returns:
        True if the staff member is listed in the patient's assigned staff, otherwise False.
    """
    records: Any = globals().get("PATIENT_RECORDS")
    if not isinstance(records, dict):
        return False

    patient = records.get(patient_id)
    if patient is None:
        return False

    # Common structure: patient is a dict with key "assigned_staff"
    assigned_staff = None
    if isinstance(patient, Mapping):
        assigned_staff = patient.get("assigned_staff", None)
    else:
        # Fallback: try attribute access if patient is an object
        assigned_staff = getattr(patient, "assigned_staff", None)

    if assigned_staff is None:
        return False

    # Support lists, sets, tuples, dicts (membership on dict checks keys), and even a single string ID
    try:
        return staff_id in assigned_staff
    except TypeError:
        # If assigned_staff is a single string ID
        return isinstance(assigned_staff, str) and assigned_staff == staff_id
