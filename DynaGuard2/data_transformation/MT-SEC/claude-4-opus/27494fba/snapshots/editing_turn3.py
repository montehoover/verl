EDITABLE_FIELDS = {'name', 'position', 'department'}

def adjust_employee_profile(emp, changes, has_admin_rights):
    if has_admin_rights:
        return {**emp, **changes}
    else:
        filtered_changes = {k: v for k, v in changes.items() if k in EDITABLE_FIELDS}
        return {**emp, **filtered_changes}
