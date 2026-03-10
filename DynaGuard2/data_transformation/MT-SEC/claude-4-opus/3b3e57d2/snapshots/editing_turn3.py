EDITABLE_FIELDS = {"name", "position", "department"}

def modify_staff_details(employee_data, changes, is_superuser):
    modified_data = employee_data.copy()
    
    for field, value in changes.items():
        if is_superuser or field in EDITABLE_FIELDS:
            modified_data[field] = value
    
    return modified_data
