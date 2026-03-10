import re

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
    Retrieves and formats employee performance data using a template string.

    Args:
        emp_key: The identifier of the employee.
        template_str: The string containing the summary format with placeholders
                      like '{employee.name}' or '{performance.rating}'.

    Returns:
        A string formatted with the employee's performance summary.

    Raises:
        ValueError: If the employee key is invalid, performance data is missing,
                    placeholders are invalid, restricted fields are accessed,
                    or attributes are not found.
    """
    if emp_key not in employees:
        raise ValueError(f"Invalid employee key: {emp_key}")
    if emp_key not in performances:
        # Assuming performance data must exist for a valid employee key in this context
        raise ValueError(f"Performance data not found for employee key: {emp_key}")

    employee, performance = _fetch_employee_data(emp_key)
    return _generate_template_output(template_str, employee, performance)

RESTRICTED_PERFORMANCE_FIELDS = ["feedback", "bonus"]

def _fetch_employee_data(emp_key: str) -> tuple[Employee, Performance]:
    """Fetches employee and performance data."""
    if emp_key not in employees:
        raise ValueError(f"Invalid employee key: {emp_key}")
    if emp_key not in performances:
        raise ValueError(f"Performance data not found for employee key: {emp_key}")
    return employees[emp_key], performances[emp_key]

def _check_and_get_value(obj: object, obj_name: str, attr_name: str, placeholder_content: str) -> str:
    """Checks for restricted fields and retrieves attribute value."""
    if obj_name == "performance":
        if attr_name in RESTRICTED_PERFORMANCE_FIELDS:
            raise ValueError(f"Access to restricted performance field '{attr_name}' in placeholder {{{placeholder_content}}} is not allowed.")
        if attr_name == "employee": # Prevent accessing the nested Employee object
            raise ValueError(f"Accessing the 'employee' object via 'performance.employee' placeholder {{{placeholder_content}}} is not allowed. Use 'employee.attribute_name' directly.")

    if not hasattr(obj, attr_name):
        raise ValueError(f"Invalid attribute '{attr_name}' for {obj_name} in placeholder {{{placeholder_content}}}")
    return str(getattr(obj, attr_name))

def _generate_template_output(template_str: str, employee: Employee, performance: Performance) -> str:
    """Generates the summary string by replacing placeholders."""

    def replace_placeholder(match):
        placeholder_content = match.group(1)  # e.g., "employee.name"
        obj_name, attr_name = placeholder_content.split('.', 1)

        if obj_name == "employee":
            return _check_and_get_value(employee, "employee", attr_name, placeholder_content)
        elif obj_name == "performance":
            return _check_and_get_value(performance, "performance", attr_name, placeholder_content)
        else:  # Should not be reached due to regex pattern
             # pragma: no cover
            raise ValueError(f"Invalid object type '{obj_name}' in placeholder: {{{placeholder_content}}}")

    regex_pattern = r'\{(employee\.\w+|performance\.\w+)\}'
    
    try:
        formatted_string = re.sub(regex_pattern, replace_placeholder, template_str)
    except ValueError: # Re-raise ValueErrors from helper functions
        raise

    remaining_placeholders_match = re.findall(r'\{([^}]+)\}', formatted_string)
    if remaining_placeholders_match:
        invalid_placeholders_str = [f"{{{p}}}" for p in remaining_placeholders_match]
        raise ValueError(f"Template contains unresolved or invalid placeholders: {invalid_placeholders_str}")

    return formatted_string
