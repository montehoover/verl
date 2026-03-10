from string import Formatter


def build_employee_summary(emp_key: str, template_str: str) -> str:
    """
    Return a formatted summary using a template string with placeholders such as
    '{employee.name}' and '{performance.rating}'.

    Requirements:
      - Raises ValueError if the employee ID is not found.
      - Raises ValueError if the template attempts to access restricted fields:
        'performance.feedback' or 'performance.bonus'.

    Expects global dictionaries:
      - `employees`: mapping employee IDs to Employee instances
      - `performances`: mapping employee IDs to Performance instances
    """
    employees_dict = globals().get("employees", {})
    if emp_key not in employees_dict:
        raise ValueError(f"Employee ID '{emp_key}' not found")
    employee = employees_dict[emp_key]

    performances_dict = globals().get("performances", {})
    performance = performances_dict.get(emp_key)

    # Validate that the template does not attempt to access restricted fields.
    formatter = Formatter()
    for _, field_name, _, _ in formatter.parse(template_str):
        if not field_name:
            continue
        if field_name.startswith("performance.feedback"):
            raise ValueError("Access to restricted field 'feedback' is not allowed")
        if field_name.startswith("performance.bonus"):
            raise ValueError("Access to restricted field 'bonus' is not allowed")

    # Perform formatting with the available objects.
    try:
        return template_str.format(employee=employee, performance=performance)
    except (KeyError, AttributeError, IndexError) as e:
        # Normalize template/data issues to ValueError for a clean API surface.
        raise ValueError(f"Invalid template or missing data: {e}")
