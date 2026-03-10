EDITABLE_FIELDS = {'name', 'position', 'department'}

def update_record(data, modifications):
    for key, value in modifications.items():
        if value is not None:
            data[key] = value
    return data

def modify_employee_data(worker, alterations, admin_privileges):
    if admin_privileges:
        for key, value in alterations.items():
            worker[key] = value
    else:
        for key, value in alterations.items():
            if key in EDITABLE_FIELDS:
                worker[key] = value
    return worker
