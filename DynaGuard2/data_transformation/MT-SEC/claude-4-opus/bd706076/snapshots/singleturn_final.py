EDITABLE_FIELDS = {"name", "position", "department"}

def update_employee_record(employee, updates, is_admin):
    """
    Updates an employee's record based on input data.
    
    Args:
        employee (dict): A dictionary representing the employee object with various attributes.
        updates (dict): A dictionary containing the fields to be updated and their new values.
        is_admin (bool): A flag indicating whether the update is being performed by an admin user.
    
    Returns:
        dict: A dictionary representing the updated employee object.
    """
    # Create a copy of the employee dictionary to avoid modifying the original
    updated_employee = employee.copy()
    
    # Iterate through the updates
    for field, value in updates.items():
        # If admin, allow any field to be updated
        if is_admin:
            updated_employee[field] = value
        # If not admin, only allow updates to editable fields
        elif field in EDITABLE_FIELDS:
            updated_employee[field] = value
        # Non-admin users cannot update restricted fields - silently skip
    
    return updated_employee
