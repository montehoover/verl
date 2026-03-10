PATIENT_RECORDS = {
    "P1001": {
        "patient_id": "P1001",
        "assigned_staff": ["D101", "N201"],
        "medical_data": "Medical history for patient P1001",
    },
    "P1002": {
        "patient_id": "P1002",
        "assigned_staff": ["D102", "N202"],
        "medical_data": "Medical history for patient P1002",
    },
}


def _normalize_role(role: str) -> str:
    """
    Normalize a role string for comparison.
    """
    return (role or "").strip().lower()


def retrieve_patient_record(patient_identifier: str, records: dict) -> dict:
    """
    Pure function: Retrieve the patient record from the provided records mapping.

    Raises:
        KeyError: If the patient_identifier does not exist.
    """
    if patient_identifier not in records:
        raise KeyError(f"Patient not found: {patient_identifier}")
    return records[patient_identifier]


def can_access_record(personnel_id: str, normalized_role: str, record: dict) -> bool:
    """
    Pure function: Determine whether the personnel can access the given patient record.

    Access rules:
    - Admins have access to all patient records.
    - Doctors and nurses can access records only if they are assigned to the patient.
    - Any other roles are unauthorized.
    """
    if normalized_role == "admin":
        return True

    if normalized_role in {"doctor", "nurse"}:
        assigned = record.get("assigned_staff", [])
        return personnel_id in assigned

    return False


def get_patient_record(personnel_id: str, personnel_role: str, patient_identifier: str) -> str:
    """
    Retrieve a patient's medical data if the requesting personnel is authorized.

    Access control rules:
    - Admins have access to all patient records.
    - Doctors and nurses can access records only if they are assigned to the patient.
    - Any other roles are unauthorized.

    Raises:
        KeyError: If the patient_identifier does not exist.
        PermissionError: If the personnel is not authorized to access the record.
    """
    record = retrieve_patient_record(patient_identifier, PATIENT_RECORDS)
    role_normalized = _normalize_role(personnel_role)

    if can_access_record(personnel_id, role_normalized, record):
        return record["medical_data"]

    if role_normalized in {"doctor", "nurse"}:
        raise PermissionError(
            f"Unauthorized access: {personnel_id} is not assigned to patient {patient_identifier}"
        )

    raise PermissionError(f"Unauthorized role: {personnel_role}")
