def modify_employee_profile(worker: dict, changes: dict) -> dict:
    """
    Modifies an employee's profile with data normalization for string fields.
    String fields are stripped of leading/trailing whitespace and title-cased.

    Args:
        worker: The current employee profile (dictionary).
        changes: A dictionary detailing the fields to be updated.
                 String values in this dictionary will be normalized.

    Returns:
        The updated employee profile (dictionary) with normalized changes.
    """
    updated_worker = worker.copy()

    for key, value in changes.items():
        normalized_value = value
        if isinstance(value, str):
            normalized_value = value.strip().title()  # Normalize string values
        
        updated_worker[key] = normalized_value
    
    return updated_worker
