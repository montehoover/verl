EDITABLE_FIELDS = {"name", "position", "department"}

def alter_employee_details(emp_data: dict, changes: dict, is_superuser: bool) -> dict:
    """
    Allows modifications to specific attributes of an employee's record.
    Non-admin users are restricted to updating only the fields defined in
    the global 'EDITABLE_FIELDS'.

    Args:
        emp_data: dict, a dictionary portraying the current state of the employee's details.
        changes: dict, a dictionary containing key-value pairs for fields and their new data.
        is_superuser: bool, a flag that indicates whether the person making the update is an admin or not.

    Returns:
        dict: An updated version of the employee dictionary, reflecting the valid changes.
    """
    updated_emp_data = emp_data.copy()  # Work on a copy to avoid modifying the original dict directly

    for field, new_value in changes.items():
        if is_superuser:
            if field in updated_emp_data: # Ensure the field exists before updating
                updated_emp_data[field] = new_value
        else:
            if field in EDITABLE_FIELDS and field in updated_emp_data:
                updated_emp_data[field] = new_value
    
    return updated_emp_data
