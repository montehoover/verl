EDITABLE_FIELDS = {"name", "position", "department"}

def update_employee_record(employee: dict, updates: dict, is_admin: bool) -> dict:
    """
    Updates an employee's record based on input data.

    Args:
        employee: A dictionary representing the employee object.
        updates: A dictionary containing the fields to be updated and their new values.
        is_admin: A flag indicating whether the update is being performed by an admin user.

    Returns:
        A dictionary representing the updated employee object.
    """
    # Create a copy of the employee record to avoid modifying the original dictionary in place.
    # This ensures the function is pure and does not have side effects on the input object.
    employee_record_copy = employee.copy()

    # Iterate over each field and its new value provided in the updates dictionary.
    for field_to_update, new_value in updates.items():
        # Check for permission to update the field.
        # An admin user can update any field.
        # A non-admin user can only update fields listed in EDITABLE_FIELDS.
        can_update_field = is_admin or field_to_update in EDITABLE_FIELDS

        # Guard clause: If the current user does not have permission to update this specific field,
        # skip to the next field in the updates dictionary.
        if not can_update_field:
            continue

        # If permission is granted, update the field in the copied employee record.
        employee_record_copy[field_to_update] = new_value
    
    # Return the modified copy of the employee record.
    return employee_record_copy
