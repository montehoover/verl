EDITABLE_FIELDS = {"name", "position", "department"}

def alter_employee_details(emp_data, changes, is_superuser):
    updated_employee = emp_data.copy()
    
    for field, new_value in changes.items():
        if is_superuser or field in EDITABLE_FIELDS:
            updated_employee[field] = new_value
    
    return updated_employee
