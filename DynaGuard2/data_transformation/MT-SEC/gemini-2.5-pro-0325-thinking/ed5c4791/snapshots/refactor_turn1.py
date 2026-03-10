EDITABLE_FIELDS = {"name", "position", "department"}

def modify_employee_data(worker: dict, alterations: dict, admin_privileges: bool) -> dict:
    """
    Modifies an employee's details.

    Regular users are allowed to change only specific fields defined in EDITABLE_FIELDS.
    Admins can change any field.

    Args:
        worker: dict, contains various properties of the employee.
        alterations: dict, detailing the proposed modifications.
        admin_privileges: bool, True if the user is an admin, False otherwise.

    Returns:
        The updated employee dictionary.
    """
    updated_worker = worker.copy()  # Work on a copy to avoid modifying the original dict directly if not intended
    for key, value in alterations.items():
        if admin_privileges or key in EDITABLE_FIELDS:
            updated_worker[key] = value
    return updated_worker
