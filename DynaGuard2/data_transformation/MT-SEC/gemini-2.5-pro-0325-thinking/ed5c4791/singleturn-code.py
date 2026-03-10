EDITABLE_FIELDS = {"name", "position", "department"}

def modify_employee_data(worker: dict, alterations: dict, admin_privileges: bool) -> dict:
    """
    Modifies an employee's details.

    Regular users are allowed to change only specific fields predefined
    within the global 'EDITABLE_FIELDS' set.

    Args:
        worker: dict, which contains various properties of the employee.
        alterations: dict, detailing the proposed modifications to that
                     employee's attributes.
        admin_privileges: bool, determines whether the user carrying out the
                          update is an admin or not.

    Returns:
        The updated employee dictionary.
    """
    updated_worker = worker.copy()  # Work on a copy to avoid modifying the original dict directly

    for key, value in alterations.items():
        if admin_privileges:
            updated_worker[key] = value
        elif key in EDITABLE_FIELDS:
            updated_worker[key] = value
        # If not admin and key not in EDITABLE_FIELDS, the change is ignored.

    return updated_worker
