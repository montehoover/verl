from typing import Iterable, Optional, Dict, Any

DEFAULT_EXCLUDE_FIELDS = {"social_security_number"}

PREFERRED_ORDER = [
    "name",
    "position",
    "department",
    "salary",
    "social_security_number",
    "manager",
    "hire_date",
    "email",
    "phone",
    "employee_id",
]


def _labelize(key: str) -> str:
    # Convert snake_case to Title Case labels, with light tweaks for common acronyms
    label = key.replace("_", " ").strip().title()
    # Fix common acronyms
    label = label.replace("Ssn", "SSN").replace("Id", "ID")
    return label


def _format_value(key: str, value: Any) -> str:
    if value is None:
        return ""
    if key == "salary":
        # Attempt to format as currency if numeric
        try:
            num = float(value)
            return f"${num:,.2f}"
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def display_employee_profile(
    employee_data: Dict[str, Any],
    exclude_fields: Optional[Iterable[str]] = None,
) -> str:
    """
    Format an employee profile into a readable multi-line string.

    Args:
        employee_data: Dictionary with employee fields (e.g., name, position, salary, department, social_security_number).
        exclude_fields: Iterable of field names to exclude from the output. By default, 'social_security_number' is excluded.

    Returns:
        A formatted string representing the employee profile.
    """
    if not isinstance(employee_data, dict):
        raise TypeError("employee_data must be a dict")

    exclude = set(exclude_fields) if exclude_fields is not None else set(DEFAULT_EXCLUDE_FIELDS)

    # Build ordered list of keys to display
    keys_in_output = []

    # Add preferred keys in order if present and not excluded
    for key in PREFERRED_ORDER:
        if key in employee_data and key not in exclude:
            val = employee_data.get(key)
            if val is not None and val != "":
                keys_in_output.append(key)

    # Add any remaining keys not in preferred order
    for key, val in employee_data.items():
        if key in exclude:
            continue
        if val is None or val == "":
            continue
        if key not in keys_in_output:
            keys_in_output.append(key)

    lines = ["Employee Profile:"]
    for key in keys_in_output:
        label = _labelize(key)
        value_str = _format_value(key, employee_data[key])
        lines.append(f"- {label}: {value_str}")

    return "\n".join(lines)
