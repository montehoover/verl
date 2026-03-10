EDITABLE_FIELDS = {"name", "position", "department"}

def modify_employee_data(worker: dict, alterations: dict, admin_privileges: bool) -> dict:
    """
    Modify an employee's details.

    Args:
        worker (dict): Employee data dict. Example keys: name, position, salary, department, social_security_number.
        alterations (dict): Proposed changes to employee attributes.
        admin_privileges (bool): If True, all fields can be modified. If False, only fields in EDITABLE_FIELDS can be modified.

    Returns:
        dict: The updated employee dictionary (a new dict; original is not mutated).

    Behavior:
        - Admins can modify or add any fields provided in `alterations`.
        - Regular users can only modify fields present in the global EDITABLE_FIELDS set.
          Any keys outside of EDITABLE_FIELDS are ignored for regular users.
    """
    if not isinstance(worker, dict):
        raise TypeError("worker must be a dict")
    if alterations is None:
        alterations = {}
    if not isinstance(alterations, dict):
        raise TypeError("alterations must be a dict")

    # Work on a copy to avoid mutating the original dict
    updated = worker.copy()

    if admin_privileges:
        # Admins can update any field
        updated.update(alterations)
        return updated

    # Regular users: only update allowed fields
    for key, value in alterations.items():
        if key in EDITABLE_FIELDS:
            updated[key] = value

    return updated
