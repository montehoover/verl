def view_patient_record(staff_id: str, staff_role: str, patient_record_id: str) -> str:
    """
    Retrieve a patient's medical record if the requester is authorized.

    Authorization rules:
    - 'admin' role can view any patient record.
    - 'doctor' and 'nurse' can view records only if their staff_id is listed in the patient's 'assigned_staff'.

    Raises:
        NameError: If PATIENT_RECORDS is not defined in the global scope.
        KeyError: If the patient_record_id does not exist.
        PermissionError: If the requester is not authorized to view the record.

    Returns:
        str: The patient's medical data.
    """
    # Safely access PATIENT_RECORDS from global scope to avoid undefined-name issues
    records = globals().get("PATIENT_RECORDS")
    if not isinstance(records, dict):
        raise NameError("PATIENT_RECORDS is not defined in the global scope")

    if patient_record_id not in records:
        raise KeyError(f"Patient record '{patient_record_id}' not found")

    record = records[patient_record_id]
    role = (staff_role or "").strip().lower()

    # Admins can view any record
    if role == "admin":
        return record["medical_data"]

    # Doctors and nurses can view records only if assigned to the patient
    if role in {"doctor", "nurse"} and staff_id in record.get("assigned_staff", []):
        return record["medical_data"]

    # All other cases are unauthorized
    raise PermissionError("Insufficient permissions to view this patient record.")
