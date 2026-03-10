"""
Changes:
- Implemented display_employee_details(emp_data) to format employee info into a readable string.
- Added prepare_employee_update(emp_data, changes) to produce a merged, non-mutating update dict.
- Added alter_employee_details(emp_data, changes, is_superuser) to apply updates with permissions.
"""

from typing import Any, Dict


def display_employee_details(emp_data: Dict[str, Any]) -> str:
    """
    Return a formatted string of the employee's details.

    Expected keys in emp_data:
      - name (str)
      - position (str)
      - salary (int | float | str convertible to float)
      - department (str)
      - social_security_number (str)

    Raises:
      - ValueError: if any required field is missing.
      - TypeError: if salary cannot be interpreted as a number.

    Example:
        data = {
            "name": "Jane Doe",
            "position": "Software Engineer",
            "salary": 120000,
            "department": "Engineering",
            "social_security_number": "123-45-6789",
        }
        print(display_employee_details(data))
    """
    required_keys = [
        "name",
        "position",
        "salary",
        "department",
        "social_security_number",
    ]
    missing = [k for k in required_keys if k not in emp_data]
    if missing:
        raise ValueError(f"Missing required employee fields: {', '.join(missing)}")

    name = str(emp_data["name"])
    position = str(emp_data["position"])
    department = str(emp_data["department"])
    ssn = str(emp_data["social_security_number"])

    salary_val = emp_data["salary"]
    if isinstance(salary_val, (int, float)):
        salary_num = float(salary_val)
    elif isinstance(salary_val, str):
        try:
            salary_num = float(salary_val.replace(",", "").strip())
        except ValueError as exc:
            raise TypeError("salary must be numeric or a numeric string") from exc
    else:
        raise TypeError("salary must be numeric or a numeric string")

    salary_formatted = f"${salary_num:,.2f}"

    details = (
        "Employee Details:\n"
        f"Name: {name}\n"
        f"Position: {position}\n"
        f"Department: {department}\n"
        f"Salary: {salary_formatted}\n"
        f"Social Security Number: {ssn}"
    )
    return details


def prepare_employee_update(emp_data: Dict[str, Any], changes: Dict[str, Any]) -> Dict[str, Any]:
    """
    Produce a new dictionary that reflects proposed updates, keeping any
    unchanged fields from emp_data. This function does not mutate inputs.

    - If a key exists in both emp_data and changes:
        - If both values are dicts, perform a recursive merge to preserve
          unchanged nested fields.
        - Otherwise, use the value from changes.
    - Keys present only in emp_data are preserved.
    - Keys present only in changes are added.

    Example:
        current = {
            "name": "Jane Doe",
            "position": "Software Engineer",
            "salary": 120000,
            "department": "Engineering",
            "social_security_number": "123-45-6789",
            "address": {"city": "Seattle", "zip": "98101"},
        }
        upd = {
            "position": "Senior Software Engineer",
            "address": {"zip": "98109"},
        }
        result = prepare_employee_update(current, upd)
        # result["position"] == "Senior Software Engineer"
        # result["address"] == {"city": "Seattle", "zip": "98109"}
    """
    def _merge(a: Any, b: Any) -> Any:
        if isinstance(a, dict) and isinstance(b, dict):
            merged: Dict[str, Any] = {}
            keys = set(a.keys()) | set(b.keys())
            for k in keys:
                if k in a and k in b:
                    merged[k] = _merge(a[k], b[k])
                elif k in b:
                    merged[k] = b[k]
                else:
                    merged[k] = a[k]
            return merged
        # For non-dict values, prefer b (the change)
        return b

    return _merge(emp_data, changes)


# Fields that non-admin users are allowed to edit
EDITABLE_FIELDS = {"name", "position", "department"}


def alter_employee_details(
    emp_data: Dict[str, Any],
    changes: Dict[str, Any],
    is_superuser: bool,
) -> Dict[str, Any]:
    """
    Apply changes to an employee record in-place while enforcing permissions.

    - Superusers can modify any field present in `changes`.
    - Non-admin users can only modify fields included in EDITABLE_FIELDS.

    Args:
        emp_data: The current employee data to be updated (modified in-place).
        changes: The proposed changes as a dict of key -> new value.
        is_superuser: If True, bypasses field restrictions.

    Returns:
        The updated `emp_data` dictionary (same object, updated in-place).

    Examples:
        alter_employee_details(emp, {"position": "Manager"}, False)
        alter_employee_details(emp, {"salary": 150000}, True)
    """
    if is_superuser:
        for key, value in changes.items():
            emp_data[key] = value
        return emp_data

    # Non-admin: only allow updates to EDITABLE_FIELDS
    for key, value in changes.items():
        if key in EDITABLE_FIELDS:
            emp_data[key] = value

    return emp_data
