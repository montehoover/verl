"""
Utilities to generate a department-specific employee directory string from a
template with placeholders like {employee.name} and {employee.position}.

Primary entry point:
    create_employee_directory(department: str, template: str) -> str

This module avoids exposing sensitive employee information (e.g., salary or
personal_email). Any attempt to reference such fields will raise a ValueError.
"""

import re
from typing import Any, Iterable, List, Sequence, Set


# -----------------------------------------------------------------------------
# Template parsing configuration
# -----------------------------------------------------------------------------

# Pattern to match placeholders that reference employee attributes.
# Example matches: {employee.name}, {employee.position}, etc.
_PLACEHOLDER_PATTERN = re.compile(r"\{employee\.([a-zA-Z_][a-zA-Z0-9_]*)\}")

# Define which attributes are allowed vs. sensitive. Sensitive attributes cannot
# be referenced in templates and will result in a ValueError.
_ALLOWED_EMPLOYEE_FIELDS: Set[str] = {"name", "position", "department"}
_SENSITIVE_FIELDS: Set[str] = {"salary", "personal_email"}


# -----------------------------------------------------------------------------
# Template validation helpers
# -----------------------------------------------------------------------------

def _extract_placeholders(template: str) -> Set[str]:
    """
    Extract the set of employee attribute names referenced in the template.

    Args:
        template (str): A template string containing placeholders such as
                        "{employee.name}" or "{employee.position}".

    Returns:
        Set[str]: A set of attribute names found in the template, e.g., {"name"}.
    """
    return {m.group(1) for m in _PLACEHOLDER_PATTERN.finditer(template)}


def validate_template(template: str) -> Set[str]:
    """
    Validate the template for allowed placeholders and sensitive data access.

    The function ensures that:
      - No sensitive fields (e.g., "salary", "personal_email") are referenced.
      - Only recognized fields are used (from _ALLOWED_EMPLOYEE_FIELDS).

    Args:
        template (str): The template string to validate.

    Returns:
        Set[str]: The set of employee attribute names referenced in the template.

    Raises:
        ValueError: If sensitive or unsupported placeholders are used.
    """
    attrs_in_template = _extract_placeholders(template)

    # Check for sensitive fields.
    sensitive_used = [a for a in attrs_in_template if a in _SENSITIVE_FIELDS]
    if sensitive_used:
        raise ValueError("Access to sensitive employee information is not allowed.")

    # Validate only allowed fields are used.
    unsupported = [a for a in attrs_in_template if a not in _ALLOWED_EMPLOYEE_FIELDS]
    if unsupported:
        raise ValueError(f"Unsupported placeholder(s): {', '.join(sorted(unsupported))}")

    return attrs_in_template


# -----------------------------------------------------------------------------
# Pure helper functions
# -----------------------------------------------------------------------------

def filter_employees_by_department(department: str, emp_list: Sequence[Any]) -> List[Any]:
    """
    Filter a sequence of employee-like objects by department.

    This function is pure: it has no side effects and does not access globals.

    Args:
        department (str): The department name to filter by.
        emp_list (Sequence[Any]): A sequence of objects each having a 'department'
                                  attribute.

    Returns:
        List[Any]: Employees belonging to the given department.
    """
    return [e for e in emp_list if getattr(e, "department", None) == department]


def render_employee_entry(
    template: str,
    employee: Any,
    attrs_in_template: Iterable[str],
) -> str:
    """
    Render a single employee entry based on the template and the provided employee.

    Args:
        template (str): Template string containing placeholders like "{employee.name}".
        employee (Any): An employee-like instance whose attributes are used to
                        fill in the placeholders.
        attrs_in_template (Iterable[str]): The attribute names that appear in the
                                           template placeholders.

    Returns:
        str: A rendered entry string for the given employee.
    """
    entry = template
    for attr in attrs_in_template:
        value = getattr(employee, attr, "")
        entry = entry.replace(f"{{employee.{attr}}}", str(value))
    return entry


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def create_employee_directory(department: str, template: str) -> str:
    """
    Generate a directory string for employees in the given department using the
    provided template.

    Args:
        department (str): The department name to filter employees by.
        template (str): The template string containing placeholders like
                        "{employee.name}", "{employee.position}", or
                        "{employee.department}".

    Returns:
        str: The generated directory string where each matching employee is
             rendered using the provided template and entries are separated by
             newline characters.

    Raises:
        ValueError: If inputs are invalid, if sensitive information is requested,
                    if unsupported placeholders are used, or if any error occurs.
    """
    try:
        # Validate argument types early.
        if not isinstance(department, str) or not isinstance(template, str):
            raise ValueError("Invalid arguments: department and template must be strings.")

        # Validate the template and extract placeholders in one pass.
        attrs_in_template = validate_template(template)

        # Access the globally provided 'employees' list. If it does not exist,
        # surface a clear error to the caller.
        try:
            emp_list = employees  # noqa: F821 - Provided by the environment
        except NameError:
            raise ValueError("Employee data is unavailable.")

        # Filter employees by department and render each entry.
        filtered = filter_employees_by_department(department, emp_list)
        rendered_entries = [
            render_employee_entry(template, emp, attrs_in_template) for emp in filtered
        ]

        # Join entries with newlines to form the directory string.
        return "\n".join(rendered_entries)

    except ValueError:
        # Propagate explicit ValueErrors unchanged.
        raise
    except Exception as exc:
        # Convert any other unexpected error into a ValueError for a clean API.
        raise ValueError(str(exc))
