from string import Formatter
from typing import Tuple, Any

# Allowed and restricted attributes
ALLOWED_EMPLOYEE_ATTRS = {"emp_id", "name", "position", "department"}
ALLOWED_PERFORMANCE_ATTRS = {"rating"}
RESTRICTED_ATTRS = {"feedback", "bonus"}

_FORMATTER = Formatter()


def fetch_employee_data(emp_key: str) -> Tuple[Any, Any]:
    """
    Fetch the Employee and Performance instances for a given employee key.

    Raises:
        ValueError: If required data sources are missing or the employee/performance is not found.
    """
    try:
        employee = employees.get(emp_key)  # type: ignore[name-defined]
        performance = performances.get(emp_key)  # type: ignore[name-defined]
    except NameError as e:
        raise ValueError("Required data sources are not available") from e

    if employee is None or performance is None:
        raise ValueError("Invalid employee ID or performance data not found")

    return employee, performance


def _iter_placeholders(template_str: str):
    """
    Yield field names found in the template string.
    """
    for _, field_name, _, _ in _FORMATTER.parse(template_str):
        if field_name:
            yield field_name


def check_restricted_fields(template_str: str) -> None:
    """
    Ensure no restricted fields are requested within the template.
    Raises:
        ValueError: If a restricted field is requested.
    """
    for field_name in _iter_placeholders(template_str):
        # Only consider dotted access patterns; other validations happen elsewhere
        parts = field_name.split(".")
        if len(parts) == 2:
            _, attr = parts[0].strip(), parts[1].strip()
            if attr in RESTRICTED_ATTRS:
                raise ValueError(f"Access to restricted field '{attr}' is not allowed")


def _validate_placeholders(template_str: str) -> None:
    """
    Validate placeholders for correct structure and allowed attributes.
    Raises:
        ValueError: If placeholders are invalid or unsupported.
    """
    for _, field_name, _, _ in _FORMATTER.parse(template_str):
        if not field_name:
            continue

        if field_name.isdigit():
            raise ValueError(f"Invalid placeholder '{field_name}'")

        parts = field_name.split(".")
        if len(parts) != 2:
            raise ValueError(f"Invalid placeholder field '{field_name}'")

        root, attr = parts[0].strip(), parts[1].strip()

        if root not in {"employee", "performance"}:
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
    Raises:
        ValueError: If formatting fails unexpectedly.
    """
    try:
        return _FORMATTER.vformat(
            template_str,
            args=(),
            kwargs={"employee": employee, "performance": performance},
        )
    except Exception as e:
        raise ValueError("Failed to format the template string") from e


def build_employee_summary(emp_key: str, template_str: str) -> str:
    """
    Build a formatted employee performance summary from a template.

    Args:
        emp_key: Employee identifier.
        template_str: Template string containing placeholders such as
                      '{employee.name}', '{employee.position}', '{performance.rating}'.

    Returns:
        A formatted summary string.

    Raises:
        ValueError: If the employee ID is invalid, if invalid data is accessed,
                    or if restricted fields like 'feedback' or 'bonus' are requested.
    """
    employee, performance = fetch_employee_data(emp_key)
    check_restricted_fields(template_str)
    _validate_placeholders(template_str)
    return generate_template_output(template_str, employee, performance)
