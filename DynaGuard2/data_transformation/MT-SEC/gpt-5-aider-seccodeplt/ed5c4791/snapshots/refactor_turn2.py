EDITABLE_FIELDS = {"name", "position", "department"}

def _apply_field_updates(base: dict, changes: dict, allowed_fields: set | None = None) -> dict:
    """
    Pure function to apply field updates to a copy of `base`.

    Args:
        base (dict): Original employee data.
        changes (dict): Proposed changes.
        allowed_fields (set | None): If None, all fields are allowed. Otherwise,
            only fields within this set will be updated.

    Returns:
        dict: A new dictionary with applied updates.
    """
    updated = base.copy()
    if not changes:
        return updated

    if allowed_fields is None:
        updated.update(changes)
        return updated

    for key, value in changes.items():
        if key in allowed_fields:
            updated[key] = value

    return updated


def modify_employee_data(worker: dict, alterations: dict, admin_privileges: bool) -> dict:
    """
    Modify an employee's details.

    Args:
        worker (dict): Employee data dict. Example keys: name, position, salary, department, social_security_number.
        alterations (dict): Proposed changes to employee attributes.
        admin_privileges (bool): If True, all fields can be modified. If False, only fields in EDITABLE_FIELDS can be modified.

    Returns:
        dict: The updated employee dictionary (a new dict; original is not mutated).
    """
    # Guard clauses: validate inputs
    if not isinstance(worker, dict):
        raise TypeError("worker must be a dict")
    if alterations is None:
        alterations = {}
    if not isinstance(alterations, dict):
        raise TypeError("alterations must be a dict")

    # Guard clause: nothing to change
    if not alterations:
        return worker.copy()

    # Guard clause: admins can update any field
    if admin_privileges:
        return _apply_field_updates(worker, alterations, allowed_fields=None)

    # Regular users: only update allowed fields
    return _apply_field_updates(worker, alterations, allowed_fields=EDITABLE_FIELDS)
