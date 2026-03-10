PATIENT_RECORDS = {
    "P1001": {
        "patient_id": "P1001",
        "assigned_staff": ["D101", "N201"], # Assuming D for Doctor, N for Nurse
        "medical_data": "Medical history for patient P1001"
    },
    "P1002": {
        "patient_id": "P1002",
        "assigned_staff": ["D102", "N202"],
        "medical_data": "Medical history for patient P1002"
    }
}


def authenticate_worker(worker_id: str, password: str) -> bool:
    """
    Authenticates a healthcare worker based on their ID and password.

    Args:
        worker_id: The worker's unique identifier.
        password: The worker's password.

    Returns:
        True if the credentials are correct, False otherwise.
    """
    # In a real application, this would involve checking credentials
    # against a secure database or authentication service.
    # For this example, we'll use a placeholder.
    # TODO: Replace with actual authentication logic.
    if worker_id == "test_worker" and password == "password123":
        return True
    return False


def check_worker_role(worker_id: str, required_role: str) -> bool:
    """
    Checks if a worker has the required role.

    Args:
        worker_id: The worker's unique identifier.
        required_role: The role required for a specific action (e.g., 'doctor', 'nurse', 'admin').

    Returns:
        True if the worker has the required role, False otherwise.
    """
    # In a real application, this would involve checking the worker's role
    # against a database or role management system.
    # For this example, we'll use a placeholder.
    # TODO: Replace with actual role checking logic.
    worker_roles = {
        "test_worker": "admin",
        "doctor_jane": "doctor",
        "nurse_john": "nurse"
    }
    if worker_id in worker_roles and worker_roles[worker_id] == required_role:
        return True
    # Admins should have access to all roles
    if worker_id in worker_roles and worker_roles[worker_id] == "admin":
        return True
    return False


def get_patient_data(worker_id: str, worker_role: str, record_id: str) -> str:
    """
    Retrieves patient medical data based on worker authorization.

    Args:
        worker_id: The worker's unique identifier.
        worker_role: The role of the worker (e.g., 'doctor', 'nurse', 'admin').
        record_id: The ID of the patient record to retrieve.

    Returns:
        The patient's medical data.

    Raises:
        ValueError: If the record_id does not exist.
        PermissionError: If the worker is not authorized to access the record.
    """
    if record_id not in PATIENT_RECORDS:
        raise ValueError(f"Record ID {record_id} not found.")

    patient_record = PATIENT_RECORDS[record_id]

    # Admins have access to all records
    if worker_role == "admin":
        return patient_record["medical_data"]

    # Doctors and Nurses can access records if they are assigned
    if worker_role in ["doctor", "nurse"]:
        if worker_id in patient_record["assigned_staff"]:
            return patient_record["medical_data"]
        else:
            raise PermissionError(
                f"Worker {worker_id} with role {worker_role} is not authorized to access record {record_id}."
            )
    
    # Other roles (if any) are not authorized by default
    raise PermissionError(
        f"Worker {worker_id} with role {worker_role} is not authorized to access record {record_id}."
    )
