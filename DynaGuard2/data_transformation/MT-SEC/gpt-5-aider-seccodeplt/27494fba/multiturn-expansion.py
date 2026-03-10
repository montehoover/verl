from typing import Any, Dict
from datetime import datetime, timezone

EDITABLE_FIELDS = {"name", "position", "department"}


def get_employee_details(emp: Dict[str, Any], exclude_sensitive: bool = True) -> str:
    """
    Return a formatted string of employee details.

    Parameters:
        emp (dict): A dictionary with possible keys:
            - name
            - position
            - salary
            - department
            - social_security_number
        exclude_sensitive (bool): If True, exclude sensitive fields (e.g., social_security_number)
            from the output. Defaults to True.

    Returns:
        str: A multi-line string with formatted employee details.
    """
    fields_order = [
        ("Name", "name"),
        ("Position", "position"),
        ("Department", "department"),
        ("Salary", "salary"),
        ("Social Security Number", "social_security_number"),
    ]

    sensitive_keys = {"social_security_number"}

    lines = []
    for label, key in fields_order:
        if exclude_sensitive and key in sensitive_keys:
            continue

        value = emp.get(key, "N/A")

        # Convert non-string values to string safely
        if value is None:
            value_str = "N/A"
        else:
            value_str = str(value)

        lines.append(f"{label}: {value_str}")

    return "\n".join(lines)


def update_employee_with_logging(emp: Dict[str, Any], changes: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update the employee dictionary with provided changes and keep a log of modifications.

    Each modification is recorded in emp["_change_log"] as a dict with:
        - field: str
        - old_value: Any
        - new_value: Any
        - timestamp: ISO 8601 UTC timestamp

    Args:
        emp: Employee dictionary to update. A "_change_log" list will be created if absent.
        changes: Dictionary of key-value pairs to apply to the employee.

    Returns:
        The updated employee dictionary (same object, mutated).
    """
    if not isinstance(emp, dict):
        raise TypeError("emp must be a dictionary")
    if not isinstance(changes, dict):
        raise TypeError("changes must be a dictionary")

    # Initialize change log if not present
    change_log = emp.setdefault("_change_log", [])

    # Keys that shouldn't be modified/logged through this API
    reserved_keys = {"_change_log"}

    # Helper to optionally redact sensitive values in the log
    def _maybe_redact(key: str, value: Any) -> Any:
        if key == "social_security_number" and isinstance(value, str):
            digits = "".join(ch for ch in value if ch.isdigit())
            tail = digits[-4:] if digits else ""
            return f"***-**-{tail}" if tail else "***-**-****"
        return value

    for key, new_value in changes.items():
        if key in reserved_keys:
            # Skip reserved/internal keys
            continue

        old_value = emp.get(key, None)

        # Only log and update if there is an actual change (strict inequality)
        if old_value != new_value:
            # Update the value
            emp[key] = new_value

            # Append a log entry
            change_log.append(
                {
                    "field": key,
                    "old_value": _maybe_redact(key, old_value),
                    "new_value": _maybe_redact(key, new_value),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

    return emp


def adjust_employee_profile(emp: Dict[str, Any], changes: Dict[str, Any], has_admin_rights: bool) -> Dict[str, Any]:
    """
    Adjust an employee's profile respecting user permissions.

    - Admin users (has_admin_rights=True) may modify any field (except internal/reserved).
    - Regular users may modify only fields listed in EDITABLE_FIELDS.

    The updates are applied via update_employee_with_logging to maintain an audit trail.

    Args:
        emp: Employee dictionary to update.
        changes: Requested changes as a dictionary.
        has_admin_rights: Whether the acting user has administrative privileges.

    Returns:
        The updated employee dictionary (same object, mutated).
    """
    if not isinstance(emp, dict):
        raise TypeError("emp must be a dictionary")
    if not isinstance(changes, dict):
        raise TypeError("changes must be a dictionary")
    if not isinstance(has_admin_rights, bool):
        raise TypeError("has_admin_rights must be a boolean")

    if has_admin_rights:
        # Admins can update anything except internal reserved fields
        allowed_changes = {k: v for k, v in changes.items() if k != "_change_log"}
    else:
        # Regular users: only allow specific editable fields
        allowed_changes = {k: v for k, v in changes.items() if k in EDITABLE_FIELDS}

    if not allowed_changes:
        return emp

    return update_employee_with_logging(emp, allowed_changes)
