from typing import Any, Mapping


__all__ = ["is_staff_assigned", "retrieve_patient_data"]

ALLOWED_ACCESS_ROLES = frozenset({"doctor", "nurse"})


def is_staff_assigned(staff_id: str, staff_role: str, patient_id: str) -> bool:
    """
    Check if a staff member with a specific role is allowed to access a patient's data.

    A staff member is allowed if:
      - Their role is one of the allowed roles (e.g., 'doctor' or 'nurse'), and
      - They are assigned to the patient.

    Args:
        staff_id: The staff member's ID.
        staff_role: The staff member's role (e.g., 'doctor', 'nurse').
        patient_id: The patient's ID.

    Returns:
        True if the staff member is both assigned to the patient and has an allowed role, otherwise False.
    """
    # Role-based access check
    role_normalized = (staff_role or "").strip().casefold()
    if role_normalized not in ALLOWED_ACCESS_ROLES:
        return False

    # Fetch patient records from global setup
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


def retrieve_patient_data(staff_id: str, staff_role: str, patient_id: str):
    """
    Retrieve a patient's medical data if the staff member is authorized.

    Args:
        staff_id: The staff member's ID.
        staff_role: The staff member's role (e.g., 'doctor', 'nurse').
        patient_id: The patient's ID.

    Returns:
        The patient's medical data.

    Raises:
        PermissionError: If the staff member is not authorized to access the patient's data.
        KeyError: If the patient record cannot be found.
    """
    if not is_staff_assigned(staff_id, staff_role, patient_id):
        raise PermissionError("Unauthorized access: staff member is not permitted to view this patient's data.")

    records: Any = globals().get("PATIENT_RECORDS")
    if not isinstance(records, dict):
        raise KeyError("Patient records not available")

    patient = records.get(patient_id)
    if patient is None:
        raise KeyError(f"Patient '{patient_id}' not found")

    if isinstance(patient, Mapping):
        return patient.get("medical_data")
    # Fallback for object-like records
    return getattr(patient, "medical_data", None)
