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
