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

    employee = employees[emp_key]
    performance = performances[emp_key]

    restricted_performance_fields = ["feedback", "bonus"]

    def replace_placeholder(match):
        placeholder_content = match.group(1)  # e.g., "employee.name" or "performance.rating"
        
        # The regex pattern ensures placeholder_content is "object.attribute"
        obj_name, attr_name = placeholder_content.split('.', 1)

        if obj_name == "employee":
            if not hasattr(employee, attr_name):
                raise ValueError(f"Invalid attribute '{attr_name}' for employee in placeholder {{{placeholder_content}}}")
            return str(getattr(employee, attr_name))
        elif obj_name == "performance":
            if attr_name in restricted_performance_fields:
                raise ValueError(f"Access to restricted performance field '{attr_name}' in placeholder {{{placeholder_content}}} is not allowed.")
            if attr_name == "employee": # Prevent accessing the nested Employee object directly as performance.employee
                raise ValueError(f"Accessing the 'employee' object via 'performance.employee' placeholder {{{placeholder_content}}} is not allowed. Use 'employee.attribute_name' directly.")
            if not hasattr(performance, attr_name):
                raise ValueError(f"Invalid attribute '{attr_name}' for performance in placeholder {{{placeholder_content}}}")
            return str(getattr(performance, attr_name))
        # The main regex pattern should prevent this else case from being reached.
        # Kept as a defensive measure.
        else: # pragma: no cover
            raise ValueError(f"Invalid object type '{obj_name}' in placeholder: {{{placeholder_content}}}")

    # Regex to find placeholders like {employee.attribute} or {performance.attribute}
    # \w+ matches one or more word characters (letters, numbers, underscore).
    regex_pattern = r'\{(employee\.\w+|performance\.\w+)\}'
    
    try:
        formatted_string = re.sub(regex_pattern, replace_placeholder, template_str)
    except ValueError: # Re-raise ValueErrors from replace_placeholder
        raise

    # After substitution, check if any other types of placeholders remain.
    # This catches placeholders not matching the specific 'employee.attr' or 'performance.attr' patterns.
    remaining_placeholders_match = re.findall(r'\{([^}]+)\}', formatted_string)
    if remaining_placeholders_match:
        # This indicates placeholders that were not in the allowed 'employee.attr' 
        # or 'performance.attr' format, or were otherwise unresolvable by the pattern.
        invalid_placeholders_str = [f"{{{p}}}" for p in remaining_placeholders_match]
        raise ValueError(f"Template contains unresolved or invalid placeholders: {invalid_placeholders_str}")

    return formatted_string
