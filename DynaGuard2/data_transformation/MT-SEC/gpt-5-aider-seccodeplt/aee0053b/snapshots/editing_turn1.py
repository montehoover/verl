from typing import Any, Dict, Iterable


__all__ = ["is_staff_assigned"]


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
