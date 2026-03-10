EDITABLE_FIELDS = {"name", "position", "department"}

def modify_employee_profile(worker: dict, changes: dict, is_privileged: bool) -> dict:
    """
    Modifies an employee's profile based on user privileges.
    String fields are stripped of leading/trailing whitespace and title-cased.

    Args:
        worker: The current employee profile (dictionary).
        changes: A dictionary detailing the fields to be updated.
                 String values in this dictionary will be normalized.
        is_privileged: A boolean indicating if the user has administrative privileges.
                       If True, all fields in `changes` can be updated.
                       If False, only fields in `EDITABLE_FIELDS` can be updated.

    Returns:
        The updated employee profile (dictionary).
    """
    updated_worker = worker.copy()

    for key, value in changes.items():
        normalized_value = value
        if isinstance(value, str):
            normalized_value = value.strip().title()  # Normalize string values
        
        if is_privileged:
            updated_worker[key] = normalized_value
        elif key in EDITABLE_FIELDS:
            updated_worker[key] = normalized_value
        # else: field is not editable by non-privileged user, so change is ignored.
    
    return updated_worker
