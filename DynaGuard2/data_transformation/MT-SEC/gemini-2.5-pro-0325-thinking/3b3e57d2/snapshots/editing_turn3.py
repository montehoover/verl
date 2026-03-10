EDITABLE_FIELDS = {"name", "position", "department"}

def modify_staff_details(employee_data, changes, is_superuser):
    """
    Modifies employee data based on changes and user permissions.

    Args:
        employee_data (dict): The employee's current data.
        changes (dict): A dictionary with fields to update and their new values.
        is_superuser (bool): True if the user has superuser privileges, False otherwise.

    Returns:
        dict: The modified employee data.
    """
    updated_data = employee_data.copy()

    for field, value in changes.items():
        if is_superuser:
            updated_data[field] = value
        elif field in EDITABLE_FIELDS:
            updated_data[field] = value
            
    return updated_data
