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


def get_patient_record(personnel_id: str, personnel_role: str, patient_identifier: str) -> str:
    """
    Retrieve a patient's medical data if access control requirements are satisfied.

    Access control rules:
    - 'admin' role: access to any patient record.
    - 'doctor' or 'nurse': access only if the personnel_id is in the patient's assigned_staff.

    Args:
        personnel_id: ID of the healthcare personnel requesting access.
        personnel_role: Role of the user (e.g., 'doctor', 'nurse', 'admin').
        patient_identifier: ID of the patient whose record is requested.

    Returns:
        The medical_data string for the patient if authorized.

    Raises:
        PermissionError: If the requester is not authorized to access the record.
        KeyError: If the patient record does not exist.
        TypeError: If provided arguments are not strings.
    """
    # Basic type validation to ensure expected usage
    if not isinstance(personnel_id, str) or not isinstance(personnel_role, str) or not isinstance(patient_identifier, str):
        raise TypeError("All arguments must be strings: personnel_id, personnel_role, patient_identifier")

    patient = PATIENT_RECORDS.get(patient_identifier)
    if patient is None:
        raise KeyError(f"Patient record not found for ID: {patient_identifier}")

    role = personnel_role.strip().lower()

    if role == "admin":
        return patient["medical_data"]

    if role in {"doctor", "nurse"} and personnel_id in patient.get("assigned_staff", []):
        return patient["medical_data"]

    raise PermissionError("Unauthorized access to patient medical data.")
