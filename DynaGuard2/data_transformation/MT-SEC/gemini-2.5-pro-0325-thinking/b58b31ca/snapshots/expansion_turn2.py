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
