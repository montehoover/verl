EDITABLE_FIELDS = {"name", "position", "department"}


def modify_employee_data(worker: dict, alterations: dict, admin_privileges: bool) -> dict:
    """
    Modify an employee's details.

    Parameters:
        worker (dict): The current employee data (e.g., name, position, salary, department, social_security_number).
        alterations (dict): Proposed changes to the employee's attributes.
        admin_privileges (bool): Whether the updater is an admin. Admins can change any field; regular users can only
                                 change fields listed in EDITABLE_FIELDS.

    Returns:
        dict: The updated employee dictionary.
    """
    if not isinstance(worker, dict):
        raise TypeError("worker must be a dict")
    if not isinstance(alterations, dict):
        raise TypeError("alterations must be a dict")

    updated = worker.copy()

    if admin_privileges:
        # Admins can modify any field.
        for key, value in alterations.items():
            updated[key] = value
        return updated

    # Regular users can only modify specific fields.
    for key, value in alterations.items():
        if key in EDITABLE_FIELDS:
            updated[key] = value

    return updated
