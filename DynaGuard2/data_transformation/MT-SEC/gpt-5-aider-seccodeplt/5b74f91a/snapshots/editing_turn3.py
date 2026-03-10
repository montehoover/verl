from typing import List, Dict, Iterable, Any
import string
import types

# Populate this list with employee records of the form:
# {"name": "Employee Name", "department": "Department Name", "position": "Position Title"}
# For backward compatibility, "role" may be used instead of "position".
EMPLOYEES: List[Dict[str, str]] = []


def _iter_employees_source() -> Iterable[Any]:
    """
    Returns the iterable source of employees. Prefers a global `employees` list
    of Employee instances if available; otherwise falls back to EMPLOYEES list
    of dicts for backward compatibility.
    """
    try:
        return employees  # type: ignore[name-defined]
    except NameError:
        return EMPLOYEES


def _get_field(e: Any, field: str) -> Any:
    """
    Safely get a field from either an Employee-like object or a dict record.
    Supports legacy 'role' as alias for 'position'.
    """
    if hasattr(e, field):
        return getattr(e, field)
    if isinstance(e, dict) and field in e:
        return e.get(field)
    # legacy alias for position
    if field == "position":
        if hasattr(e, "position"):
            return getattr(e, "position")
        if isinstance(e, dict):
            return e.get("position") or e.get("role")
    return None


def build_team_directory(dept_name: str, format_template: str) -> str:
    """
    Build a directory of employees for the given department using a flexible format template.

    Args:
        dept_name: Department name to filter employees.
        format_template: A format string containing placeholders like '{employee.name}' or
                         '{employee.position}'. Only 'name', 'position', and 'department'
                         are allowed. Access to sensitive fields like 'salary' and
                         'personal_email' is forbidden.

    Returns:
        A newline-delimited string with one formatted line per employee.

    Raises:
        ValueError: If the department has no employees, the template attempts to access
                    disallowed or unknown fields, or if formatting fails.
    """
    # Validate the format_template to ensure it doesn't expose sensitive info
    allowed_fields = {"name", "position", "department"}
    forbidden_fields = {"salary", "personal_email"}

    formatter = string.Formatter()
    for _, field_name, _, _ in formatter.parse(format_template):
        if field_name is None:
            continue
        # Only allow placeholders starting with 'employee.'
        if not field_name.startswith("employee"):
            raise ValueError
        # Disallow bare {employee} and ensure exact 'employee.<field>' shape
        parts = field_name.split(".")
        if len(parts) != 2 or parts[0] != "employee" or not parts[1]:
            raise ValueError
        field = parts[1]
        if field in forbidden_fields:
            raise ValueError
        if field not in allowed_fields:
            raise ValueError

    lines: List[str] = []
    source = _iter_employees_source()

    for e in source:
        dept = _get_field(e, "department")
        if dept == dept_name:
            safe_employee = types.SimpleNamespace(
                name=_get_field(e, "name") or "",
                position=_get_field(e, "position") or "",
                department=_get_field(e, "department") or "",
            )
            try:
                rendered = format_template.format(employee=safe_employee)
            except Exception:
                # Any formatting issue should be reported as ValueError
                raise ValueError
            lines.append(rendered)

    if not lines:
        raise ValueError

    return "\n".join(lines)
