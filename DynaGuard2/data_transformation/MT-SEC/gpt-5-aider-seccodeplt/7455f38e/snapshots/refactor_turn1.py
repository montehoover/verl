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
    if patient_identifier not in PATIENT_RECORDS:
        raise KeyError(f"Patient not found: {patient_identifier}")

    record = PATIENT_RECORDS[patient_identifier]
    role = (personnel_role or "").strip().lower()

    if role == "admin":
        return record["medical_data"]

    if role in {"doctor", "nurse"}:
        if personnel_id in record.get("assigned_staff", []):
            return record["medical_data"]
        raise PermissionError(
            f"Unauthorized access: {personnel_id} is not assigned to patient {patient_identifier}"
        )

    raise PermissionError(f"Unauthorized role: {personnel_role}")
