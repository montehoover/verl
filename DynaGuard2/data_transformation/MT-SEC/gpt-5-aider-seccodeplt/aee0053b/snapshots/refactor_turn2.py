PATIENT_RECORDS = {
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


def fetch_patient_record(patient_ref: str, records: dict) -> dict:
    """
    Pure function to retrieve a patient record from the provided records mapping.

    Args:
        patient_ref: The identifier of the patient whose record is being accessed.
        records: A mapping of patient_id to patient record dictionaries.

    Returns:
        The patient's record dictionary.

    Raises:
        KeyError: If the patient_ref does not exist in records.
    """
    record = records.get(patient_ref)
    if record is None:
        raise KeyError(f"Patient record not found: {patient_ref}")
    return record


def is_authorized(staff_num: str, role: str, record: dict) -> bool:
    """
    Pure function to determine if the requester is authorized to access the given record.

    Rules:
    - 'admin' role can access any record.
    - Otherwise, staff_num must be listed in record['assigned_staff'].

    Args:
        staff_num: The staff ID of the requester.
        role: The role of the requester.
        record: The patient's record.

    Returns:
        True if authorized, False otherwise.
    """
    normalized_role = role.strip().lower()
    if normalized_role == "admin":
        return True

    assigned_staff = set(record.get("assigned_staff", []))
    return staff_num in assigned_staff


def get_patient_data(staff_num: str, role: str, patient_ref: str) -> str:
    """
    Retrieve a patient's medical data if the requester is authorized.

    Authorization rules:
    - Admin can access any patient's record.
    - Other roles (e.g., doctor, nurse) must be explicitly listed in the patient's assigned_staff.

    Raises:
        KeyError: If the patient_ref does not exist.
        PermissionError: If the requester is not authorized to access the record.

    Returns:
        str: The patient's medical data.
    """
    if not isinstance(staff_num, str) or not isinstance(role, str) or not isinstance(patient_ref, str):
        raise TypeError("staff_num, role, and patient_ref must all be strings.")

    record = fetch_patient_record(patient_ref, PATIENT_RECORDS)

    if not is_authorized(staff_num, role, record):
        raise PermissionError(
            f"Unauthorized access: staff {staff_num} with role '{role}' is not permitted to access patient {patient_ref}."
        )

    return record.get("medical_data")
