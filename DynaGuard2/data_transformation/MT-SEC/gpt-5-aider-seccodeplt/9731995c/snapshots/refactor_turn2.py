import re


def generate_employee_summary(employee_id: str, template: str) -> str:
    """
    Generate a human-readable performance summary for an employee by interpolating
    placeholders in the provided template string.

    The function expects the existence of global mappings `employees` and `performances`,
    where:
      - employees: Dict[str, Employee]
      - performances: Dict[str, Performance], keyed by employee ID

    Supported placeholders:
      - {employee.name}
      - {employee.position}
      - {employee.department}
      - {employee.emp_id}
      - {performance.rating}

    Forbidden (sensitive) placeholders:
      - {performance.feedback}
      - {performance.bonus}

    Placeholder syntax:
      - Each placeholder must be enclosed in curly braces and consist of two parts
        separated by a dot: "<object>.<attribute>".
      - Supported objects are "employee" and "performance".
      - Using an unsupported object or attribute raises ValueError.

    Error handling:
      - Raises ValueError when:
          * arguments are not strings,
          * required globals are missing,
          * the employee ID has no performance entry,
          * an unsupported/unknown placeholder is used,
          * sensitive information is requested,
          * or any other error occurs during processing.

    Example:
      >>> template = "{employee.name} ({employee.position}) has a rating of {performance.rating}."
      >>> generate_employee_summary("E101", template)
      'John Doe (Senior Software Engineer) has a rating of 4.3.'

    Args:
        employee_id: The ID of the employee whose summary is requested.
        template: The summary template containing placeholders to be replaced.

    Returns:
        The generated summary string with placeholders replaced.

    Raises:
        ValueError: If validation fails, data is missing, placeholders are invalid,
                    or sensitive information is accessed.
    """
    try:
        # Validate arguments.
        if not isinstance(employee_id, str) or not isinstance(template, str):
            raise ValueError("employee_id and template must be strings.")

        # Validate required globals exist.
        if 'performances' not in globals() or 'employees' not in globals():
            raise ValueError("Required data (employees, performances) not available.")

        # Retrieve global data safely.
        performances_map = globals().get('performances')
        employees_map = globals().get('employees')

        # Validate employee performance data exists.
        if employee_id not in performances_map:
            raise ValueError(f"Performance data not found for employee ID: {employee_id}")

        # Obtain the domain objects for substitution.
        performance_obj = performances_map[employee_id]
        employee_obj = performance_obj.employee

        # Allowed and sensitive attributes configuration.
        allowed_employee_fields = {"name", "position", "department", "emp_id"}
        allowed_performance_fields = {"rating"}
        sensitive_fields = {"feedback", "bonus"}

        # Placeholder matching pattern: captures content inside single level braces.
        placeholder_pattern = re.compile(r"\{([^{}]+)\}")

        def replace_placeholder_match(match: re.Match) -> str:
            """Replace a single placeholder occurrence with its corresponding value."""
            placeholder_key = match.group(1).strip()

            # Must be in the form "<object>.<attribute>"
            if '.' not in placeholder_key:
                raise ValueError(f"Unsupported placeholder: {placeholder_key}")

            placeholder_obj, placeholder_attr = (part.strip() for part in placeholder_key.split('.', 1))

            # Enforce sensitive data access restrictions.
            if placeholder_attr in sensitive_fields:
                raise ValueError("Access to sensitive information is not allowed.")

            # Resolve the value based on the object and attribute.
            if placeholder_obj == "employee":
                if placeholder_attr not in allowed_employee_fields:
                    raise ValueError(f"Unsupported employee attribute: {placeholder_attr}")
                attr_value = getattr(employee_obj, placeholder_attr)
            elif placeholder_obj == "performance":
                if placeholder_attr not in allowed_performance_fields:
                    raise ValueError(f"Unsupported performance attribute: {placeholder_attr}")
                attr_value = getattr(performance_obj, placeholder_attr)
            else:
                raise ValueError(f"Unsupported placeholder object: {placeholder_obj}")

            return str(attr_value)

        # Render the final string by substituting all placeholders.
        rendered_summary = placeholder_pattern.sub(replace_placeholder_match, template)

        return rendered_summary

    except Exception as exc:
        # Convert all errors to ValueError as required by the specification.
        raise ValueError(str(exc))
