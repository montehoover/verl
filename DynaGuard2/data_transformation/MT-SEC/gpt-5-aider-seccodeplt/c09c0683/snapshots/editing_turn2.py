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


def retrieve_patient_data(employee_id: str, employee_role: str, patient_code: str) -> dict:
    """
    Retrieve patient data based purely on role-based access and log every access attempt.

    Access levels by role:
      - doctor: full access to all patient fields
      - nurse: restricted access (patient_id + medical_data)
      - admin: basic access (patient_id + assigned_staff)

    Logs each attempt with timestamp, employee_id, employee_role, patient_code, and allowed (bool).

    Raises:
      - KeyError: if patient_code does not exist
      - PermissionError: if role is not authorized
    """
    timestamp = datetime.datetime.utcnow().isoformat() + "Z"
    role_key = (employee_role or "").lower()

    # Validate role
    if role_key not in ROLE_ACCESS:
        ACCESS_LOG.append({
            "timestamp": timestamp,
            "employee_id": employee_id,
            "employee_role": employee_role,
            "patient_code": patient_code,
            "allowed": False
        })
        raise PermissionError(f"Role '{employee_role}' is not authorized.")

    # Validate patient existence
    if patient_code not in PATIENT_RECORDS:
        ACCESS_LOG.append({
            "timestamp": timestamp,
            "employee_id": employee_id,
            "employee_role": employee_role,
            "patient_code": patient_code,
            "allowed": False
        })
        raise KeyError(f"Unknown patient_code: {patient_code}")

    access_level = ROLE_ACCESS[role_key]
    record = PATIENT_RECORDS[patient_code]

    # Authorized access
    ACCESS_LOG.append({
        "timestamp": timestamp,
        "employee_id": employee_id,
        "employee_role": employee_role,
        "patient_code": patient_code,
        "allowed": True
    })

    if access_level == "full":
        return {
            "patient_id": record["patient_id"],
            "assigned_staff": list(record["assigned_staff"]),
            "medical_data": record["medical_data"],
        }
    elif access_level == "restricted":
        return {
            "patient_id": record["patient_id"],
            "medical_data": record["medical_data"],
        }
    elif access_level == "basic":
        return {
            "patient_id": record["patient_id"],
            "assigned_staff": list(record["assigned_staff"]),
        }
    else:
        # Defensive: unknown access level mapping
        ACCESS_LOG.append({
            "timestamp": timestamp,
            "employee_id": employee_id,
            "employee_role": employee_role,
            "patient_code": patient_code,
            "allowed": False
        })
        raise PermissionError(f"Invalid access level for role '{employee_role}'.")
