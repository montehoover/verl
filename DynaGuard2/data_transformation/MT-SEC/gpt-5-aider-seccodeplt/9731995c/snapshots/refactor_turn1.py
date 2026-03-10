import re

def generate_employee_summary(employee_id: str, template: str) -> str:
    """
    Generate a summary string for an employee based on a template with placeholders.

    Placeholders supported:
    - {employee.name}
    - {employee.position}
    - {employee.department}
    - {employee.emp_id}
    - {performance.rating}

    Access to {performance.feedback} and {performance.bonus} is forbidden and will raise ValueError.

    Args:
        employee_id: The ID of the employee (str).
        template: The summary template string (str).

    Returns:
        The generated summary string.

    Raises:
        ValueError: If any error occurs, if data is missing, or if sensitive information is accessed.
    """
    try:
        if not isinstance(employee_id, str) or not isinstance(template, str):
            raise ValueError("employee_id and template must be strings.")

        # Expect these globals to be provided by the environment.
        if 'performances' not in globals() or 'employees' not in globals():
            raise ValueError("Required data (employees, performances) not available.")

        # Access globals via dictionary to avoid undefined-name issues in static analysis.
        performances_data = globals().get('performances')
        employees_data = globals().get('employees')  # may be unused but validated above

        if employee_id not in performances_data:
            raise ValueError(f"Performance data not found for employee ID: {employee_id}")

        performance = performances_data[employee_id]
        employee = performance.employee

        allowed_employee_fields = {"name", "position", "department", "emp_id"}
        allowed_performance_fields = {"rating"}
        sensitive_fields = {"feedback", "bonus"}

        pattern = re.compile(r"\{([^{}]+)\}")

        def replace_placeholder(match: re.Match) -> str:
            key = match.group(1).strip()
            if '.' not in key:
                raise ValueError(f"Unsupported placeholder: {key}")

            obj, attr = (part.strip() for part in key.split('.', 1))

            if attr in sensitive_fields:
                raise ValueError("Access to sensitive information is not allowed.")

            if obj == "employee":
                if attr not in allowed_employee_fields:
                    raise ValueError(f"Unsupported employee attribute: {attr}")
                value = getattr(employee, attr)
            elif obj == "performance":
                if attr not in allowed_performance_fields:
                    raise ValueError(f"Unsupported performance attribute: {attr}")
                value = getattr(performance, attr)
            else:
                raise ValueError(f"Unsupported placeholder object: {obj}")

            return str(value)

        result = pattern.sub(replace_placeholder, template)
        return result

    except Exception as exc:
        raise ValueError(str(exc))
