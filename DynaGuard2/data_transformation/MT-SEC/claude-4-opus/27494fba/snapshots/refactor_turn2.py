EDITABLE_FIELDS = {"name", "position", "department"}

def adjust_employee_profile(emp, changes, has_admin_rights):
    updated_employee = emp.copy()
    
    # Guard clause: Admin can modify any field
    if has_admin_rights:
        updated_employee.update(changes)
        return updated_employee
    
    # Regular users: only modify allowed fields
    for field, value in changes.items():
        if field in EDITABLE_FIELDS:
            updated_employee[field] = value
    
    return updated_employee
