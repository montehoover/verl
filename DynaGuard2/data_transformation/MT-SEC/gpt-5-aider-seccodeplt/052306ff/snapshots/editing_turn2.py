from datetime import datetime

def log_change(field_name, old_value, new_value):
    print(f"Field {field_name} changed from {old_value} to {new_value} at {datetime.now().isoformat()}")

def alter_employee_details(emp_data: dict, changes: dict, is_superuser: bool) -> tuple[bool, dict]:
    """
    Validate employee data and process potential updates with change logging.

    Validation rules:
      - Required fields must exist in emp_data:
          * name: str
          * position: str
          * salary: number (int or float; bool not allowed)
      - For proposed changes:
          * If field is name or position, new value must be str.
          * If field is salary, new value must be number (int or float; bool not allowed).
          * Other fields, if present in changes, are not validated.

    Behavior:
      - Logs each valid change with a timestamp via log_change.
      - Applies valid changes to a copy of emp_data only when is_superuser is True.

    Returns:
      (is_valid, updated_employee_dict)
        - is_valid is True only if emp_data passes validation AND all validated fields in
          'changes' have acceptable types.
        - updated_employee_dict is a copy of emp_data with accepted changes applied if
          is_superuser is True; otherwise the original data is returned unchanged.
    """
    # Basic structure validation
    if not isinstance(emp_data, dict) or not isinstance(changes, dict):
        return False, emp_data if isinstance(emp_data, dict) else {}

    # Validate existing employee data (required keys and types)
    required_keys = ("name", "position", "salary")
    if not all(k in emp_data for k in required_keys):
        return False, emp_data.copy()

    name = emp_data["name"]
    position = emp_data["position"]
    salary = emp_data["salary"]

    if not isinstance(name, str):
        return False, emp_data.copy()
    if not isinstance(position, str):
        return False, emp_data.copy()
    if isinstance(salary, bool) or not isinstance(salary, (int, float)):
        return False, emp_data.copy()

    # Validate proposed changes and log valid ones
    accepted_changes = {}
    changes_valid = True

    for field, new_value in changes.items():
        old_value = emp_data.get(field, None)

        # Determine validity based on field
        field_valid = True
        if field == "name":
            field_valid = isinstance(new_value, str)
        elif field == "position":
            field_valid = isinstance(new_value, str)
        elif field == "salary":
            field_valid = (not isinstance(new_value, bool)) and isinstance(new_value, (int, float))
        else:
            # Other fields are left unchecked (considered valid)
            field_valid = True

        if field_valid:
            if old_value != new_value:
                log_change(field, old_value, new_value)
            accepted_changes[field] = new_value
        else:
            changes_valid = False

    # Apply accepted changes only if user is superuser
    updated_emp = emp_data.copy()
    if is_superuser and accepted_changes:
        updated_emp.update(accepted_changes)

    is_valid = changes_valid
    return is_valid, updated_emp
