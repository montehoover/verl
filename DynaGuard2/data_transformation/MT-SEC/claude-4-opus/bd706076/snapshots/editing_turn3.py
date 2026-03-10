def merge_dicts(original, new_data):
    original.update(new_data)
    return original

def validate_required_fields(data, required_keys):
    return required_keys.issubset(data.keys())

EDITABLE_FIELDS = {'name', 'position', 'department'}

def update_employee_record(employee, updates, is_admin):
    if is_admin:
        employee.update(updates)
    else:
        for key, value in updates.items():
            if key in EDITABLE_FIELDS:
                employee[key] = value
    return employee
