EDITABLE_FIELDS = {"name", "position", "department"}

def modify_staff_details(employee_data, changes, is_superuser):
    # Create a copy of the employee data to avoid modifying the original
    updated_employee = employee_data.copy()
    
    # If superuser, apply all changes
    if is_superuser:
        updated_employee.update(changes)
    else:
        # Regular user can only modify fields in EDITABLE_FIELDS
        for field, value in changes.items():
            if field in EDITABLE_FIELDS:
                updated_employee[field] = value
    
    return updated_employee
