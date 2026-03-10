EDITABLE_FIELDS = {"name", "position", "department"}

def modify_staff_details(employee_data, changes, is_superuser):
    # Create a copy of the employee data to avoid modifying the original
    updated_employee = employee_data.copy()
    
    # Guard clause: if superuser, apply all changes and return early
    if is_superuser:
        updated_employee.update(changes)
        return updated_employee
    
    # Guard clause: for regular users, filter and apply only allowed changes
    for field, value in changes.items():
        if field not in EDITABLE_FIELDS:
            continue
        updated_employee[field] = value
    
    return updated_employee
