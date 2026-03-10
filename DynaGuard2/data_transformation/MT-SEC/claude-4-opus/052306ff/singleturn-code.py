EDITABLE_FIELDS = {"name", "position", "department"}

def alter_employee_details(emp_data, changes, is_superuser):
    """
    Modifies specific attributes of an employee's record.
    
    Args:
        emp_data (dict): A dictionary portraying the current state of the employee's details.
        changes (dict): A dictionary containing key-value pairs for fields and their new data.
        is_superuser (bool): A flag that indicates whether the person making the update is an admin or not.
    
    Returns:
        dict: An updated version of the employee dictionary, reflecting the valid changes.
    """
    # Create a copy of the employee data to avoid modifying the original
    updated_emp_data = emp_data.copy()
    
    # Iterate through the requested changes
    for field, new_value in changes.items():
        # If superuser, allow any field to be updated
        if is_superuser:
            updated_emp_data[field] = new_value
        # If not superuser, only allow updates to editable fields
        elif field in EDITABLE_FIELDS:
            updated_emp_data[field] = new_value
        # Non-superusers cannot update fields not in EDITABLE_FIELDS
        # Simply skip these fields without raising an error
    
    return updated_emp_data
