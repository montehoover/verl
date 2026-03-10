EDITABLE_FIELDS = {"name", "position", "department"}

def update_employee_record(employee, updates, is_admin):
    # Create a copy of the employee dictionary to avoid modifying the original
    updated_employee = employee.copy()
    
    # Iterate through the updates
    for field, value in updates.items():
        # Skip if non-admin tries to update a restricted field
        if not is_admin and field not in EDITABLE_FIELDS:
            continue
            
        # Skip if the field doesn't exist in the employee record
        if field not in updated_employee:
            continue
            
        # Update the field
        updated_employee[field] = value
    
    return updated_employee
