EDITABLE_FIELDS = {'name', 'position', 'department'}

def update_record(data, modifications):
    merged = dict(data)
    for key, value in modifications.items():
        if value is not None:
            merged[key] = value
    return merged

def modify_employee_data(worker, alterations, admin_privileges):
    updated = dict(worker)
    if admin_privileges:
        for key, value in alterations.items():
            updated[key] = value
    else:
        for key, value in alterations.items():
            if key in EDITABLE_FIELDS:
                updated[key] = value
    return updated
