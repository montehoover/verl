from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Tuple

EDITABLE_FIELDS = {"name", "position", "department"}

def _format_currency(value: Any) -> str:
    """
    Format a numeric value as currency with a dollar sign and two decimals.
    Returns 'N/A' if the value is missing or empty.
    """
    if value is None:
        return "N/A"
    try:
        if isinstance(value, str):
            v = value.strip().replace(",", "").replace("$", "")
            if not v:
                return "N/A"
            amount = Decimal(v)
        else:
            amount = Decimal(str(value))
        return f"${amount:,.2f}"
    except (InvalidOperation, ValueError):
        return str(value)

def display_employee_profile(profile: Dict[str, Any], exclude_sensitive: bool = True) -> str:
    """
    Build a formatted string describing an employee profile.

    Args:
        profile: A dict that may include keys: 'name', 'position', 'salary', 'department', 'social_security_number'.
        exclude_sensitive: If True (default), omit sensitive fields like 'social_security_number' from output.

    Returns:
        A multi-line string with the formatted profile.
    """
    if not isinstance(profile, dict):
        raise ValueError("profile must be a dict")

    name = profile.get("name") or "N/A"
    position = profile.get("position") or "N/A"
    salary = _format_currency(profile.get("salary"))
    department = profile.get("department") or "N/A"
    ssn = profile.get("social_security_number")

    lines = [
        "Employee Profile",
        f"Name: {name}",
        f"Position: {position}",
        f"Salary: {salary}",
        f"Department: {department}",
    ]

    if not exclude_sensitive:
        lines.append(f"Social Security Number: {ssn if ssn else 'N/A'}")

    return "\n".join(lines)

def update_and_log_profile(profile: Dict[str, Any], modifications: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Update an employee profile with provided modifications and record a log of changes.

    This function mutates the given 'profile' dictionary in place.

    Args:
        profile: The original employee profile dictionary to be updated.
        modifications: A dictionary of fields and their new values.

    Returns:
        A tuple of:
            - The updated profile dictionary.
            - A list of change log entries. Each entry is a dict:
              {"field": str, "old_value": Any, "new_value": Any}
              Only actual changes (old_value != new_value) are logged.
    """
    if not isinstance(profile, dict):
        raise ValueError("profile must be a dict")
    if not isinstance(modifications, dict):
        raise ValueError("modifications must be a dict")

    change_log: List[Dict[str, Any]] = []

    for field, new_value in modifications.items():
        old_value = profile.get(field)
        if old_value != new_value:
            profile[field] = new_value
            change_log.append({
                "field": field,
                "old_value": old_value,
                "new_value": new_value,
            })

    return profile, change_log

def edit_profile_data(profile: Dict[str, Any], modifications: Dict[str, Any], is_superuser: bool) -> Dict[str, Any]:
    """
    Edit an employee profile with role-based permissions.

    This function mutates the given 'profile' dictionary in place.

    Args:
        profile: The employee profile dictionary to be updated.
        modifications: A dictionary of proposed field updates.
        is_superuser: If True, all fields are editable. If False, only fields in EDITABLE_FIELDS are editable.

    Returns:
        The updated profile dictionary.
    """
    if not isinstance(profile, dict):
        raise ValueError("profile must be a dict")
    if not isinstance(modifications, dict):
        raise ValueError("modifications must be a dict")
    if not isinstance(is_superuser, bool):
        raise ValueError("is_superuser must be a bool")

    if is_superuser:
        allowed_mods = modifications
    else:
        allowed_mods = {k: v for k, v in modifications.items() if k in EDITABLE_FIELDS}

    for field, new_value in allowed_mods.items():
        if profile.get(field) != new_value:
            profile[field] = new_value

    return profile
