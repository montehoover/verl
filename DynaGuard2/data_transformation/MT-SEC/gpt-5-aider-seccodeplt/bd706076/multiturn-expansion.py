from typing import Any, Dict
from datetime import datetime

EDITABLE_FIELDS = {"name", "position", "department"}


def get_employee_details(employee: Dict[str, Any], exclude_sensitive: bool = True) -> str:
    """
    Return a formatted string of employee details.

    Parameters:
        employee (dict): Dictionary with keys like:
            - name
            - position
            - salary
            - department
            - social_security_number
        exclude_sensitive (bool): If True, omit sensitive fields (e.g., social_security_number)
                                  from the output. Defaults to True.

    Returns:
        str: Formatted employee details.
    """
    def _format(value: Any, default: str = "N/A") -> str:
        if value is None:
            return default
        return str(value)

    name = _format(employee.get("name"))
    position = _format(employee.get("position"))
    salary = _format(employee.get("salary"))
    department = _format(employee.get("department"))
    ssn = _format(employee.get("social_security_number"))

    lines = [
        f"Name: {name}",
        f"Position: {position}",
        f"Department: {department}",
        f"Salary: {salary}",
    ]

    if not exclude_sensitive and employee.get("social_security_number") is not None:
        lines.append(f"Social Security Number: {ssn}")

    return "\n".join(lines)


def modify_employee_with_logging(employee: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply updates to an employee dictionary and log each change with before/after values.

    Notes:
        - The change log is stored in employee['change_log'] as a list of entries.
        - Each entry contains: field, before, after, timestamp (UTC ISO 8601).
        - Updates to the 'change_log' field itself are ignored to protect the log.

    Parameters:
        employee (dict): The employee record to update. Will be modified in place.
        updates (dict): Key/value pairs to apply to the employee.

    Returns:
        dict: The updated employee dictionary (with an updated 'change_log').
    """
    change_log = employee.setdefault("change_log", [])

    for key, new_value in updates.items():
        if key == "change_log":
            # Protect the log from being directly modified via updates
            continue

        old_value = employee.get(key, None)

        # Only record a change if the value actually changes (handles new keys too)
        if old_value != new_value:
            employee[key] = new_value
            change_log.append(
                {
                    "field": key,
                    "before": old_value,
                    "after": new_value,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                }
            )

    return employee


def update_employee_record(employee: Dict[str, Any], updates: Dict[str, Any], is_admin: bool) -> Dict[str, Any]:
    """
    Update an employee record with role-based permissions.

    - Admin users can update any field (except 'change_log', which is protected).
    - Regular users can only modify fields specified in EDITABLE_FIELDS.
    - All applied changes are logged with before/after values.

    Parameters:
        employee (dict): The employee record to update (modified in place).
        updates (dict): The desired changes to apply.
        is_admin (bool): Whether the caller has admin privileges.

    Returns:
        dict: The updated employee dictionary.
    """
    if not isinstance(updates, dict):
        return employee

    if is_admin:
        filtered_updates = {k: v for k, v in updates.items() if k != "change_log"}
    else:
        filtered_updates = {k: v for k, v in updates.items() if k in EDITABLE_FIELDS}

    if not filtered_updates:
        return employee

    return modify_employee_with_logging(employee, filtered_updates)
