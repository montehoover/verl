from decimal import Decimal, InvalidOperation
from typing import Any, Dict

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
