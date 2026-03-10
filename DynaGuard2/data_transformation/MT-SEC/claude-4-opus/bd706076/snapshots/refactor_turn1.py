EDITABLE_FIELDS = {"name", "position", "department"}

def update_employee_record(employee, updates, is_admin):
    # Create a copy of the employee dictionary to avoid modifying the original
    updated_employee = employee.copy()
    
    # Iterate through the updates
    for field, value in updates.items():
        # Check if the field can be updated
        if is_admin or field in EDITABLE_FIELDS:
            # Update the field if it exists in the employee record
            if field in updated_employee:
                updated_employee[field] = value
    
    return updated_employee
