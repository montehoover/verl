"""
Utilities to build employee performance summaries using template strings.

This module provides a clean separation of concerns:
- Data fetching from global stores.
- Validation and restriction checks for template placeholders.
- Rendering of the final summary output from a template string.
"""

from string import Formatter
from typing import Any, Iterator, Tuple

# Allowed and restricted attributes (policy constraints)
ALLOWED_EMPLOYEE_ATTRS = {"emp_id", "name", "position", "department"}
ALLOWED_PERFORMANCE_ATTRS = {"rating"}
RESTRICTED_ATTRS = {"feedback", "bonus"}
ALLOWED_ROOTS = {"employee", "performance"}

# Single shared formatter instance for parsing and rendering templates
_FORMATTER = Formatter()


def fetch_employee_data(emp_key: str) -> Tuple[Any, Any]:
    """
    Fetch the Employee and Performance instances for a given employee key.

    Args:
        emp_key (str): The identifier of the employee (e.g., "E101").

    Returns:
        Tuple[Any, Any]: A tuple containing:
            - Employee: The corresponding employee object.
            - Performance: The corresponding performance object.

    Raises:
        ValueError: If required data sources are missing or the employee/performance
            records cannot be found for the provided key.
    """
    # The environment is expected to provide these globals.
    # We guard access so that the function fails with a clear error if they
    # are not defined in the hosting runtime.
    try:
        employee = employees.get(emp_key)  # type: ignore[name-defined]
        performance = performances.get(emp_key)  # type: ignore[name-defined]
    except NameError as exc:
        raise ValueError("Required data sources are not available") from exc

    if employee is None or performance is None:
        raise ValueError("Invalid employee ID or performance data not found")

    return employee, performance


def _iter_placeholders(template_str: str) -> Iterator[str]:
    """
    Yield placeholder field names found in the template string.

    Args:
        template_str (str): The template string to inspect.

    Returns:
        Iterator[str]: An iterator over placeholder field names (e.g., "employee.name").
    """
    # Formatter.parse yields tuples of: (literal_text, field_name, format_spec, conversion)
    for _, field_name, _, _ in _FORMATTER.parse(template_str):
        if field_name:
            yield field_name


def check_restricted_fields(template_str: str) -> None:
    """
    Ensure no restricted fields are requested within the template string.

    Args:
        template_str (str): The template string to validate.

    Returns:
        None

    Raises:
        ValueError: If a restricted field (e.g., "feedback" or "bonus") is requested.
    """
    for field_name in _iter_placeholders(template_str):
        # Only consider dotted access patterns; other validations happen elsewhere.
        parts = field_name.split(".")
        if len(parts) == 2:
            _, attr = parts[0].strip(), parts[1].strip()
            if attr in RESTRICTED_ATTRS:
                raise ValueError(f"Access to restricted field '{attr}' is not allowed")


def _validate_placeholders(template_str: str) -> None:
    """
    Validate placeholders for correct structure and allowed attributes.

    Args:
        template_str (str): The template string that contains placeholders.

    Returns:
        None

    Raises:
        ValueError: If placeholders are malformed, unsupported, or attempt to access
            attributes beyond the allowed set.
    """
    for _, field_name, _, _ in _FORMATTER.parse(template_str):
        if not field_name:
            continue

        # Disallow numeric/positional fields; only named fields are supported.
        if field_name.isdigit():
            raise ValueError(f"Invalid placeholder '{field_name}'")

        # We only support exactly two-level dotted access: 'root.attr'
        parts = field_name.split(".")
        if len(parts) != 2:
            raise ValueError(f"Invalid placeholder field '{field_name}'")

        root, attr = parts[0].strip(), parts[1].strip()

        if root not in ALLOWED_ROOTS:
            raise ValueError(f"Invalid placeholder root '{root}'")

        if root == "employee":
            if attr not in ALLOWED_EMPLOYEE_ATTRS:
                raise ValueError(f"Invalid employee attribute '{attr}'")
        else:  # root == "performance"
            if attr not in ALLOWED_PERFORMANCE_ATTRS:
                raise ValueError(f"Invalid performance attribute '{attr}'")


def generate_template_output(template_str: str, employee: Any, performance: Any) -> str:
    """
    Render the template with the provided employee and performance data.

    Args:
        template_str (str): The template string containing placeholders.
        employee (Any): The employee object to expose to the template.
        performance (Any): The performance object to expose to the template.

    Returns:
        str: The rendered/expanded template string.

    Raises:
        ValueError: If formatting fails unexpectedly.
    """
    # We rely on prior validation to ensure only allowed fields are present.
    # Using vformat allows us to provide a controlled mapping for named fields.
    try:
        return _FORMATTER.vformat(
            template_str,
            args=(),
            kwargs={"employee": employee, "performance": performance},
        )
    except Exception as exc:
        # Normalize any formatting error into a ValueError for a consistent API.
        raise ValueError("Failed to format the template string") from exc


def build_employee_summary(emp_key: str, template_str: str) -> str:
    """
    Build a formatted employee performance summary from a template.

    Args:
        emp_key (str): The identifier of the employee record to retrieve.
        template_str (str): The template string containing placeholders such as
            '{employee.name}', '{employee.position}', and '{performance.rating}'.

    Returns:
        str: A string formatted with the employee's performance summary.

    Raises:
        ValueError: If the employee ID is invalid, if invalid data is accessed,
            or if restricted fields (like 'feedback' or 'bonus') are requested.
    """
    # 1) Retrieve data sources for the employee
    employee, performance = fetch_employee_data(emp_key)

    # 2) Enforce policy restrictions on requested fields
    check_restricted_fields(template_str)

    # 3) Validate placeholder structure and allowed attributes
    _validate_placeholders(template_str)

    # 4) Render the final summary
    return generate_template_output(template_str, employee, performance)
