EDITABLE_FIELDS = {"name", "position", "department"}

def alter_employee_details(emp_data: dict, changes: dict, is_superuser: bool) -> tuple[bool, dict]:
    """
    Update employee details.

    - Admins (is_superuser=True) can update any fields present in 'changes'.
    - Non-admins can only update fields in EDITABLE_FIELDS.

    Returns:
      (applied_any_change, updated_employee_dict)
    """
    if not isinstance(emp_data, dict) or not isinstance(changes, dict):
        return False, emp_data if isinstance(emp_data, dict) else {}

    updated_emp = emp_data.copy()
    applied = False

    if is_superuser:
        for k, v in changes.items():
            if updated_emp.get(k) != v:
                applied = True
            updated_emp[k] = v
    else:
        for k, v in changes.items():
            if k in EDITABLE_FIELDS:
                if updated_emp.get(k) != v:
                    applied = True
                updated_emp[k] = v

    return applied, updated_emp
