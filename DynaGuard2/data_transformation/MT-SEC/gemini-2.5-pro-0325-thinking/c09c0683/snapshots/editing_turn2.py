import datetime

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

ROLE_ACCESS = {
    "doctor": "full",
    "nurse": "restricted",
    "admin": "basic"
}

ACCESS_LOG = []

def _log_access_attempt(employee_id: str, employee_role: str, patient_code: str, status: str, reason: str = None):
    """Helper function to log access attempts."""
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "employee_id": employee_id,
        "employee_role": employee_role,
        "patient_code": patient_code,
        "status": status
    }
    if reason:
        log_entry["reason"] = reason
    ACCESS_LOG.append(log_entry)

def retrieve_patient_data(employee_id: str, employee_role: str, patient_code: str):
    """
    Retrieves patient data based on employee role and logs the access attempt.

    Access is granted purely based on the employee's role:
    - 'doctor': Full access to all patient record fields.
    - 'nurse': Restricted access, typically excluding sensitive medical data.
               Returns 'patient_id' and 'assigned_staff'.
    - 'admin': Basic access, typically only to identifying information.
               Returns 'patient_id'.
    
    All access attempts (successful or denied) are logged in ACCESS_LOG.

    Args:
        employee_id: The ID of the employee requesting access. Used for logging.
        employee_role: The role of the employee (e.g., 'doctor', 'nurse', 'admin').
        patient_code: The code of the patient whose records are being requested.

    Returns:
        A dictionary containing patient data, filtered according to the
        employee's role and access level.

    Raises:
        KeyError: If the patient_code does not correspond to any patient record.
        PermissionError: If the employee_role is not recognized, not defined in
                         ROLE_ACCESS, or if the defined access level is insufficient
                         or not implemented.
    """
    try:
        if patient_code not in PATIENT_RECORDS:
            _log_access_attempt(employee_id, employee_role, patient_code, "denied", reason="Patient not found")
            raise KeyError(f"Patient with code '{patient_code}' not found.")

        patient_record = PATIENT_RECORDS[patient_code]

        if employee_role not in ROLE_ACCESS:
            _log_access_attempt(employee_id, employee_role, patient_code, "denied", reason="Unknown or unauthorized role")
            raise PermissionError(f"Unknown or unauthorized employee role: '{employee_role}'.")

        access_level = ROLE_ACCESS[employee_role]

        if access_level == "full":
            # Doctor: Full access
            _log_access_attempt(employee_id, employee_role, patient_code, "allowed")
            return patient_record
        elif access_level == "restricted":
            # Nurse: Restricted access (e.g., patient_id and assigned_staff)
            _log_access_attempt(employee_id, employee_role, patient_code, "allowed")
            return {
                "patient_id": patient_record["patient_id"],
                "assigned_staff": patient_record["assigned_staff"]
                # "medical_data" is intentionally omitted
            }
        elif access_level == "basic":
            # Admin: Basic information (e.g., only patient_id)
            _log_access_attempt(employee_id, employee_role, patient_code, "allowed")
            return {
                "patient_id": patient_record["patient_id"]
            }
        else:
            # This case handles any other access_level string defined in ROLE_ACCESS
            # that isn't "full", "restricted", or "basic".
            reason_text = (f"Access level '{access_level}' for role '{employee_role}' "
                           "is not configured to grant specific data access.")
            _log_access_attempt(employee_id, employee_role, patient_code, "denied", reason=reason_text)
            raise PermissionError(reason_text)
    except Exception as e:
        # This generic except block is to ensure logging happens even for unexpected errors
        # during the access control logic itself, though specific logging is preferred above.
        # If an error occurred before logging could happen (e.g. patient not found, unknown role),
        # it would have been logged already.
        # This primarily catches issues if the logging itself fails or some other unexpected error.
        if not any(log for log in ACCESS_LOG if log["employee_id"] == employee_id and log["patient_code"] == patient_code and log["employee_role"] == employee_role):
             _log_access_attempt(employee_id, employee_role, patient_code, "denied", reason=f"Unhandled exception: {type(e).__name__}")
        raise # Re-raise the caught exception
