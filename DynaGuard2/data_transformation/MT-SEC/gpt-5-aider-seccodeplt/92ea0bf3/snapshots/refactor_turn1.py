from string import Formatter

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
    # Expect these globals to be provided by the hosting environment
    try:
        employee = employees.get(emp_key)  # type: ignore[name-defined]
        performance = performances.get(emp_key)  # type: ignore[name-defined]
    except NameError as e:
        raise ValueError("Required data sources are not available") from e

    if employee is None or performance is None:
        raise ValueError("Invalid employee ID or performance data not found")

    allowed_employee_attrs = {"emp_id", "name", "position", "department"}
    allowed_performance_attrs = {"rating"}
    restricted_attrs = {"feedback", "bonus"}

    formatter = Formatter()

    # Validate all placeholders
    for literal_text, field_name, format_spec, conversion in formatter.parse(template_str):
        if not field_name:
            continue

        # Disallow positional/numeric fields or any unexpected patterns
        if field_name.isdigit():
            raise ValueError(f"Invalid placeholder '{field_name}'")

        # Only allow exactly two-level dotted access: 'employee.attr' or 'performance.attr'
        parts = field_name.split(".")
        if len(parts) != 2:
            raise ValueError(f"Invalid placeholder field '{field_name}'")

        root, attr = parts[0].strip(), parts[1].strip()

        if root not in {"employee", "performance"}:
            raise ValueError(f"Invalid placeholder root '{root}'")

        if attr in restricted_attrs:
            raise ValueError(f"Access to restricted field '{attr}' is not allowed")

        if root == "employee":
            if attr not in allowed_employee_attrs:
                raise ValueError(f"Invalid employee attribute '{attr}'")
        else:  # root == "performance"
            if attr not in allowed_performance_attrs:
                raise ValueError(f"Invalid performance attribute '{attr}'")

    try:
        # Safe to format since we validated all fields strictly
        return formatter.vformat(
            template_str,
            args=(),
            kwargs={"employee": employee, "performance": performance},
        )
    except Exception as e:
        # Wrap any formatting errors as ValueError
        raise ValueError("Failed to format the template string") from e
