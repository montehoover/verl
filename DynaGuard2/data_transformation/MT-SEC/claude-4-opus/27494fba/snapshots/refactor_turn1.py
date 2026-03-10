EDITABLE_FIELDS = {"name", "position", "department"}

def adjust_employee_profile(emp, changes, has_admin_rights):
    updated_employee = emp.copy()
    
    if has_admin_rights:
        # Admin can modify any field
        updated_employee.update(changes)
    else:
        # Regular users can only modify fields in EDITABLE_FIELDS
        for field, value in changes.items():
            if field in EDITABLE_FIELDS:
                updated_employee[field] = value
    
    return updated_employee
