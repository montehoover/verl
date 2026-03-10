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


def authenticate_worker(worker_id: str, password: str) -> bool:
    """
    Authenticates a healthcare worker based on their ID and password.
    
    Args:
        worker_id (str): The worker's unique identifier
        password (str): The worker's password
        
    Returns:
        bool: True if credentials are valid, False otherwise
    """
    # This is a placeholder implementation. In production, you would:
    # - Hash the password before comparing
    # - Query a secure database
    # - Use proper authentication mechanisms
    
    # Temporary hardcoded credentials for demonstration
    valid_credentials = {
        "HW001": "securepass123",
        "HW002": "healthpass456",
        "HW003": "medicalauth789"
    }
    
    # Check if worker_id exists and password matches
    if worker_id in valid_credentials:
        return valid_credentials[worker_id] == password
    
    return False


def check_worker_role(worker_id: str, required_role: str) -> bool:
    """
    Checks if a healthcare worker has the required role for an action.
    
    Args:
        worker_id (str): The worker's unique identifier
        required_role (str): The role required for the action
        
    Returns:
        bool: True if worker has the required role, False otherwise
    """
    # This is a placeholder implementation. In production, you would:
    # - Query a database for role information
    # - Use a proper role management system
    # - Consider role hierarchies
    
    # Temporary hardcoded roles for demonstration
    worker_roles = {
        "HW001": ["doctor", "admin"],
        "HW002": ["nurse"],
        "HW003": ["doctor"]
    }
    
    # Check if worker exists and has the required role
    if worker_id in worker_roles:
        return required_role in worker_roles[worker_id]
    
    return False


def get_patient_data(worker_id: str, worker_role: str, record_id: str) -> str:
    """
    Retrieves patient medical data if the worker is authorized.
    
    Args:
        worker_id (str): The worker's unique identifier
        worker_role (str): The worker's role
        record_id (str): The patient record identifier
        
    Returns:
        str: The patient's medical data if authorized
        
    Raises:
        PermissionError: If the worker is not authorized to access the record
    """
    # Check if the record exists
    if record_id not in PATIENT_RECORDS:
        raise PermissionError(f"Record {record_id} not found")
    
    patient_record = PATIENT_RECORDS[record_id]
    
    # Check if worker has admin role (admins can access all records)
    if worker_role == "admin":
        return patient_record["medical_data"]
    
    # Check if worker is assigned to this patient
    if worker_id in patient_record["assigned_staff"]:
        return patient_record["medical_data"]
    
    # If not authorized, raise PermissionError
    raise PermissionError(f"Worker {worker_id} is not authorized to access record {record_id}")
