from string import Formatter
from types import SimpleNamespace
from typing import Set


def _extract_and_validate_fields(format_template: str) -> Set[str]:
    """
    Parse the format template and validate that only allowed employee fields are referenced.
    Returns the set of employee attributes referenced.
    Raises ValueError on any disallowed or invalid placeholder usage.
    """
    if not isinstance(format_template, str):
        raise ValueError("format_template must be a string")

    formatter = Formatter()
    used_attrs: Set[str] = set()
    sensitive_fields = {"salary", "personal_email"}
    allowed_fields = {"name", "position", "department"}

    for _, field_name, _, _ in formatter.parse(format_template):
        if field_name is None:
            continue  # literal chunk, no placeholder
        # Disallow empty/positional fields like '{}' or index-based '{0}'
        if not field_name or field_name.isdigit():
            raise ValueError(f"Invalid or unsupported placeholder: {{{field_name}}}")

        # Disallow indexing, calls, or complex expressions
        if any(ch in field_name for ch in "[]()"):
            raise ValueError(f"Unsupported placeholder syntax: {{{field_name}}}")

        # Expect only 'employee.<attr>' pattern
        parts = field_name.split(".")
        if parts[0] != "employee":
            raise ValueError(f"Unsupported placeholder: {{{field_name}}}")
        if len(parts) != 2:
            raise ValueError(f"Unsupported nested attribute path: {{{field_name}}}")

        attr = parts[1]

        if attr in sensitive_fields:
            raise ValueError(f"Access to sensitive field '{attr}' is not allowed")

        if attr not in allowed_fields:
            raise ValueError(f"Unknown or disallowed employee attribute: '{attr}'")

        used_attrs.add(attr)

    return used_attrs


def _get_employees_from_globals():
    """
    Retrieve the global 'employees' list.
    Raises ValueError if not present.
    """
    emps = globals().get("employees")
    if emps is None:
        raise ValueError("Global 'employees' is not defined")
    return emps


def _filter_employees_by_department(emps, dept_name: str):
    """
    Pure function: return employees whose department matches dept_name.
    """
    return [emp for emp in emps if getattr(emp, "department", None) == dept_name]


def _to_safe_namespace(emp) -> SimpleNamespace:
    """
    Map an Employee to a safe namespace exposing only allowed attributes.
    """
    return SimpleNamespace(
        name=getattr(emp, "name", ""),
        position=getattr(emp, "position", ""),
        department=getattr(emp, "department", ""),
    )


def _format_employee_entry(emp, format_template: str) -> str:
    """
    Pure function: format a single employee using the provided template.
    Expects that the template has been pre-validated.
    """
    safe_emp = _to_safe_namespace(emp)
    try:
        return format_template.format(employee=safe_emp)
    except Exception as exc:
        raise ValueError(f"Error formatting template: {exc}") from exc


def _format_directory(emps, format_template: str) -> str:
    """
    Pure function: format a list of employees into a newline-joined directory string.
    """
    lines = [_format_employee_entry(emp, format_template) for emp in emps]
    return "\n".join(lines)


def build_team_directory(dept_name: str, format_template: str) -> str:
    """
    Build a directory listing for employees in the specified department,
    formatting each employee entry using the given template.

    - dept_name: str - department to include
    - format_template: str - template string, placeholders like '{employee.name}', '{employee.position}'

    Returns:
        str: concatenated formatted entries (one per matching employee, separated by newlines)

    Raises:
        ValueError: if the template attempts to access sensitive data (salary, personal_email),
                    uses unsupported placeholders, or any formatting error occurs.
    """
    try:
        if not isinstance(dept_name, str):
            raise ValueError("dept_name must be a string")

        # Validate template fields before formatting
        _extract_and_validate_fields(format_template)

        # Retrieve employees and filter by department
        emps = _get_employees_from_globals()
        matching = _filter_employees_by_department(emps, dept_name)

        # Format directory
        return _format_directory(matching, format_template)
    except ValueError:
        # Re-raise ValueError unchanged
        raise
    except Exception as exc:
        # Convert any unexpected error into ValueError as required
        raise ValueError(f"An error occurred while building the directory: {exc}") from exc
