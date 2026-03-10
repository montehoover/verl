EDITABLE_FIELDS = {'name', 'position', 'department'}

def update_record(data: dict, modifications: dict) -> dict:
    """
    Updates a dictionary with key-value pairs from another dictionary.

    Args:
        data: The original dictionary.
        modifications: A dictionary containing keys and values to update in data.

    Returns:
        The updated dictionary.
    """
    for key, value in modifications.items():
        if value is not None:
            data[key] = value
    return data

def modify_employee_data(worker: dict, alterations: dict, admin_privileges: bool) -> dict:
    """
    Modifies employee data based on user permissions.

    Args:
        worker: Dictionary of employee details.
        alterations: Dictionary of proposed modifications.
        admin_privileges: Boolean indicating user permissions.

    Returns:
        The updated employee dictionary.
    """
    for key, value in alterations.items():
        if value is None:
            continue  # Skip null values

        if admin_privileges:
            worker[key] = value
        else:
            if key in EDITABLE_FIELDS:
                worker[key] = value
    return worker
