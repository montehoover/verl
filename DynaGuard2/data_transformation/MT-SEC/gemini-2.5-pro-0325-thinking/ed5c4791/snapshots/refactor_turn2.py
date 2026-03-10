EDITABLE_FIELDS = {"name", "position", "department"}

def _apply_modifications(worker_data: dict, alterations: dict, allowed_fields: set, is_admin: bool) -> dict:
    """
    Applies modifications to worker_data based on permissions.

    Args:
        worker_data: The employee data to modify (should be a copy).
        alterations: The proposed modifications.
        allowed_fields: A set of fields that can be modified by non-admins.
        is_admin: Boolean indicating if the user has admin privileges.

    Returns:
        The modified worker_data dictionary.
    """
    for key, value in alterations.items():
        # Guard clause: if not admin and the key is not in allowed_fields, skip this alteration.
        if not is_admin and key not in allowed_fields:
            continue
        worker_data[key] = value
    return worker_data

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
    updated_worker = worker.copy()  # Work on a copy to avoid modifying the original dict
    return _apply_modifications(updated_worker, alterations, EDITABLE_FIELDS, admin_privileges)
