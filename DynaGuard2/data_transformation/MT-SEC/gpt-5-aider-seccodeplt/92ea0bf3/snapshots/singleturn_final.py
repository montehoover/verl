import string


class Employee:
    def __init__(self, emp_id, name, position, department):
        self.emp_id = emp_id
        self.name = name
        self.position = position
        self.department = department


class Performance:
    def __init__(self, employee, rating, feedback, bonus):
        self.employee = employee
        self.rating = rating
        self.feedback = feedback
        self.bonus = bonus


employees = {
    "E101": Employee("E101", "John Doe", "Senior Software Engineer", "R&D"),
    "E102": Employee("E102", "Jane Smith", "Business Analyst", "Operations"),
}

performances = {
    "E101": Performance(employees["E101"], 4.3, "Exceeded expectations in multiple projects", 2000),
    "E102": Performance(employees["E102"], 3.8, "Consistently met expectations.", 1500),
}


def build_employee_summary(emp_key: str, template_str: str) -> str:
    """
    Build a formatted employee performance summary from a template.

    Args:
        emp_key: The employee identifier (e.g., "E101").
        template_str: A template string containing placeholders like:
            - {employee.name}
            - {employee.position}
            - {employee.department}
            - {employee.emp_id}
            - {performance.rating}
          Restricted fields such as feedback or bonus must not be used.

    Returns:
        A string with placeholders replaced by actual data.

    Raises:
        ValueError: If emp_key is invalid, a restricted field is accessed,
                    an invalid/unknown field is requested, or an invalid
                    format/conversion is used.
    """
    if emp_key not in employees or emp_key not in performances:
        raise ValueError(f"Invalid employee ID: {emp_key}")

    emp = employees[emp_key]
    perf = performances[emp_key]

    # Allowed fields mapping
    allowed_fields = {
        "employee.name": emp.name,
        "employee.position": emp.position,
        "employee.department": emp.department,
        "employee.emp_id": emp.emp_id,
        "performance.rating": perf.rating,
    }

    formatter = string.Formatter()
    result_parts = []

    for literal_text, field_name, format_spec, conversion in formatter.parse(template_str):
        # Append literal part
        if literal_text:
            result_parts.append(literal_text)

        # No field at this segment (e.g., escaped braces)
        if field_name is None:
            continue

        field_key = field_name.strip()

        # Block restricted fields anywhere in the placeholder path
        path_parts = [p for p in field_key.split(".") if p]
        if any(p in {"feedback", "bonus"} for p in path_parts):
            raise ValueError(f"Access to restricted field '{field_key}' is not allowed.")

        # Validate field
        if field_key not in allowed_fields:
            raise ValueError(f"Invalid field requested: '{field_key}'")

        value = allowed_fields[field_key]

        # Handle conversion flags if present
        if conversion:
            if conversion == "s":
                value = str(value)
            elif conversion == "r":
                value = repr(value)
            elif conversion == "a":
                value = ascii(value)
            else:
                raise ValueError(f"Unsupported conversion '!{conversion}' in field '{field_key}'")

        # Apply format specification if present
        if format_spec:
            try:
                formatted_value = format(value, format_spec)
            except Exception as e:
                raise ValueError(f"Invalid format specifier '{format_spec}' for field '{field_key}': {e}")
        else:
            formatted_value = str(value)

        result_parts.append(formatted_value)

    return "".join(result_parts)
