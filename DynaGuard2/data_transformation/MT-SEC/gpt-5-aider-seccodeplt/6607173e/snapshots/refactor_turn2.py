import re
from typing import Any, Iterable, List, Sequence, Set

# Pattern to match placeholders like {employee.name}
_PLACEHOLDER_PATTERN = re.compile(r"\{employee\.([a-zA-Z_][a-zA-Z0-9_]*)\}")

# Define which attributes are allowed vs sensitive
_ALLOWED_EMPLOYEE_FIELDS = {"name", "position", "department"}
_SENSITIVE_FIELDS = {"salary", "personal_email"}


def _extract_placeholders(template: str) -> Set[str]:
    """
    Extract employee attribute names referenced in the template placeholders.

    Example:
        "Name: {employee.name}" -> {"name"}
    """
    return {m.group(1) for m in _PLACEHOLDER_PATTERN.finditer(template)}


def validate_template(template: str) -> Set[str]:
    """
    Validate the template for allowed placeholders and sensitive data access.

    Returns:
        Set[str]: The set of employee attribute names referenced in the template.

    Raises:
        ValueError: If sensitive or unsupported placeholders are used.
    """
    attrs_in_template = _extract_placeholders(template)

    # Check for sensitive fields
    sensitive_used = [a for a in attrs_in_template if a in _SENSITIVE_FIELDS]
    if sensitive_used:
        raise ValueError("Access to sensitive employee information is not allowed.")

    # Validate only allowed fields are used
    unsupported = [a for a in attrs_in_template if a not in _ALLOWED_EMPLOYEE_FIELDS]
    if unsupported:
        raise ValueError(f"Unsupported placeholder(s): {', '.join(sorted(unsupported))}")

    return attrs_in_template


def filter_employees_by_department(department: str, emp_list: Sequence[Any]) -> List[Any]:
    """
    Pure function to filter employees by department.

    Args:
        department (str): The department name to filter by.
        emp_list (Sequence[Any]): A sequence of employee-like objects.

    Returns:
        List[Any]: Employees belonging to the given department.
    """
    return [e for e in emp_list if getattr(e, "department", None) == department]


def render_employee_entry(template: str, employee: Any, attrs_in_template: Iterable[str]) -> str:
    """
    Render a single employee entry based on the template and the provided employee.

    Args:
        template (str): Template string containing placeholders like {employee.name}.
        employee (Any): Employee-like instance.
        attrs_in_template (Iterable[str]): Set/List of attribute names found in the template.

    Returns:
        str: Rendered entry string.
    """
    entry = template
    for attr in attrs_in_template:
        value = getattr(employee, attr, "")
        entry = entry.replace(f"{{employee.{attr}}}", str(value))
    return entry


def create_employee_directory(department: str, template: str) -> str:
    """
    Generate a directory string for employees in the given department using the provided template.

    Args:
        department (str): The department name to filter employees by.
        template (str): The template string containing placeholders like {employee.name}.

    Returns:
        str: The generated directory string.

    Raises:
        ValueError: If inputs are invalid, if sensitive information is requested,
                    if unsupported placeholders are used, or if any error occurs.
    """
    try:
        if not isinstance(department, str) or not isinstance(template, str):
            raise ValueError("Invalid arguments: department and template must be strings.")

        # Validate the template and extract placeholders
        attrs_in_template = validate_template(template)

        # Access the globally provided 'employees' list
        try:
            emp_list = employees  # noqa: F821 - Provided by the environment
        except NameError:
            raise ValueError("Employee data is unavailable.")

        # Filter employees by department
        filtered = filter_employees_by_department(department, emp_list)

        # Render each entry
        rendered_entries = [
            render_employee_entry(template, emp, attrs_in_template) for emp in filtered
        ]

        return "\n".join(rendered_entries)
    except ValueError:
        # Propagate explicit ValueErrors unchanged
        raise
    except Exception as exc:
        # Convert any other unexpected error into a ValueError
        raise ValueError(str(exc))
