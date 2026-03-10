from string import Formatter
from types import SimpleNamespace
from typing import Any, Iterable, List, Set


def _extract_and_validate_fields(format_template: str) -> Set[str]:
    """Validate placeholders in the template and collect referenced attributes.

    This function parses the provided format template and ensures that only
    non-sensitive, explicitly allowed fields are referenced using the pattern
    '{employee.<attr>}'.

    Sensitive fields such as 'salary' and 'personal_email' are explicitly blocked.
    Any unsupported placeholder syntax or attempt to access disallowed data will
    result in a ValueError.

    Args:
        format_template: The template string to validate.

    Returns:
        A set of employee attribute names referenced in the template.

    Raises:
        ValueError: If the template is not a string, contains invalid or unsupported
            placeholders, or attempts to access sensitive or unknown attributes.
    """
    # Guard: ensure template is a string early to avoid attribute errors later.
    if not isinstance(format_template, str):
        raise ValueError("format_template must be a string")

    formatter = Formatter()
    used_attrs: Set[str] = set()

    # Define sensitive and allowed fields. Sensitive fields must never be exposed.
    sensitive_fields = {"salary", "personal_email"}
    allowed_fields = {"name", "position", "department"}

    # Parse the template to inspect placeholder fields.
    for _, field_name, _, _ in formatter.parse(format_template):
        if field_name is None:
            # Literal text segment; nothing to validate.
            continue

        # Guard: Disallow empty or positional placeholders like '{}' or '{0}'.
        if not field_name or field_name.isdigit():
            raise ValueError(f"Invalid or unsupported placeholder: {{{field_name}}}")

        # Guard: Disallow indexing, function calls, or complex expressions.
        if any(ch in field_name for ch in "[]()"):
            raise ValueError(f"Unsupported placeholder syntax: {{{field_name}}}")

        # Expect exactly 'employee.<attr>' with a single attribute access.
        parts = field_name.split(".")
        if parts[0] != "employee":
            raise ValueError(f"Unsupported placeholder: {{{field_name}}}")
        if len(parts) != 2:
            raise ValueError(f"Unsupported nested attribute path: {{{field_name}}}")

        attr = parts[1]

        # Guard: Block access to sensitive fields up front.
        if attr in sensitive_fields:
            raise ValueError(f"Access to sensitive field '{attr}' is not allowed")

        # Guard: Only allow known safe fields.
        if attr not in allowed_fields:
            raise ValueError(f"Unknown or disallowed employee attribute: '{attr}'")

        used_attrs.add(attr)

    return used_attrs


def _get_employees_from_globals():
    """Retrieve the global 'employees' list.

    Returns:
        The global employees collection.

    Raises:
        ValueError: If the global 'employees' is not available.
    """
    # Guard: Fail fast if the expected global is not present.
    emps = globals().get("employees")
    if emps is None:
        raise ValueError("Global 'employees' is not defined")
    return emps


def _filter_employees_by_department(emps: Iterable[Any], dept_name: str) -> List[Any]:
    """Filter employees by department.

    Args:
        emps: An iterable of employee-like objects.
        dept_name: Department name used for filtering.

    Returns:
        A list of employees belonging to the specified department.
    """
    # Pure filtering: no mutation, no dependency on global state.
    return [emp for emp in emps if getattr(emp, "department", None) == dept_name]


def _to_safe_namespace(emp: Any) -> SimpleNamespace:
    """Create a safe proxy with only allowed attributes from an Employee.

    This proxy is used to protect against accidental exposure of sensitive fields
    during string formatting. Only 'name', 'position', and 'department' are exposed.

    Args:
        emp: The employee-like object.

    Returns:
        A SimpleNamespace exposing only safe attributes.
    """
    # Note: Sensitive attributes such as salary or personal_email are intentionally
    # omitted and therefore cannot be accessed from templates.
    return SimpleNamespace(
        name=getattr(emp, "name", ""),
        position=getattr(emp, "position", ""),
        department=getattr(emp, "department", ""),
    )


def _format_employee_entry(emp: Any, format_template: str) -> str:
    """Format a single employee using the provided, pre-validated template.

    Template errors are handled here by catching exceptions from str.format
    and converting them into a ValueError, maintaining a consistent error type.

    Args:
        emp: The employee-like object to format.
        format_template: The previously validated template string.

    Returns:
        A formatted string for the given employee.

    Raises:
        ValueError: If formatting fails for any reason.
    """
    # Use the safe namespace to ensure templates cannot reach sensitive data.
    safe_emp = _to_safe_namespace(emp)

    try:
        # Any KeyError/AttributeError raised here will be caught and wrapped below.
        return format_template.format(employee=safe_emp)
    except Exception as exc:
        # Guard: Surface template formatting issues clearly to the caller.
        # This is the central place where template rendering problems are managed.
        raise ValueError(f"Error formatting template: {exc}") from exc


def _format_directory(emps: Iterable[Any], format_template: str) -> str:
    """Format a sequence of employees into a newline-joined directory string.

    Args:
        emps: The employees to format.
        format_template: The template string used for each employee entry.

    Returns:
        A newline-separated string of formatted employee entries.
    """
    # Map each employee to its formatted representation, then join with newlines.
    lines = [_format_employee_entry(emp, format_template) for emp in emps]
    return "\n".join(lines)


def build_team_directory(dept_name: str, format_template: str) -> str:
    """Build a formatted directory listing for a specified department.

    This function orchestrates the overall flow:
    - Validates the input template to ensure no sensitive data is referenced and
      only supported placeholders are used.
    - Retrieves the global employees collection.
    - Filters employees by the requested department.
    - Formats the filtered employees using the template, capturing formatting errors
      as ValueError for a consistent error handling surface.

    Args:
        dept_name: The department whose employees to include.
        format_template: The template string with placeholders like
            '{employee.name}' or '{employee.position}'.

    Returns:
        A formatted directory string containing one line per employee in the
        specified department.

    Raises:
        ValueError: If input types are invalid, if the template references
            sensitive or unsupported fields, if global data is missing, or
            if any formatting error occurs.
    """
    try:
        # Guard: Validate the type of the department name early.
        if not isinstance(dept_name, str):
            raise ValueError("dept_name must be a string")

        # Guard: Validate the template for allowed placeholders and block sensitive data.
        _extract_and_validate_fields(format_template)

        # Retrieve employees and filter by department; fail fast if global is missing.
        emps = _get_employees_from_globals()
        matching = _filter_employees_by_department(emps, dept_name)

        # Format directory using only safe attributes; rendering errors are wrapped as ValueError.
        return _format_directory(matching, format_template)

    except ValueError:
        # Re-raise ValueError unchanged to preserve the contract described above.
        raise

    except Exception as exc:
        # Guard: Convert any unexpected error into a ValueError for consistent error handling.
        raise ValueError(f"An error occurred while building the directory: {exc}") from exc
