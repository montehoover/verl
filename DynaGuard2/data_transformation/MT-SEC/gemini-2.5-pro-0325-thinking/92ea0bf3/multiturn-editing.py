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
    Generates a summary string for an employee based on a template.

    Args:
        emp_key: The employee ID.
        template_str: A format string with placeholders for employee and performance attributes.
                      Example: "Name: {employee.name}, Rating: {performance.rating}"

    Returns:
        A formatted string with employee information.

    Raises:
        ValueError: If the employee ID is not found, or if the template attempts
                    to access restricted fields (i.e., 'performance.feedback' 
                    or 'performance.bonus', including their sub-attributes).
        AttributeError: If the template string attempts to access a non-existent
                        attribute on the employee or performance objects (and it's not
                        a restricted field).
    """
    if emp_key not in employees:
        raise ValueError(f"Employee ID {emp_key} not found.")

    employee = employees[emp_key]
    # Assuming performance data exists for every valid employee key as per current data structure.
    # If 'performances' could be missing for an employee, 'performances.get(emp_key)' might be safer,
    # and then handling of a None performance object would be needed if the template uses it.
    if emp_key not in performances:
        # This case implies data inconsistency if an employee exists but their performance record doesn't.
        # Depending on requirements, this could raise an error or provide a default/empty Performance object.
        # For now, let's assume if employee exists, performance record also exists.
        performance = None # Or handle as an error: raise ValueError(f"Performance data for {emp_key} not found.")
    else:
        performance = performances[emp_key]


    # Define base paths of restricted fields
    restricted_base_fields = {"performance.feedback", "performance.bonus"}
    
    formatter = string.Formatter()
    # Extract all unique field names requested by the template
    parsed_field_names = {fn for _, fn, _, _ in formatter.parse(template_str) if fn is not None}

    for field_name in parsed_field_names:
        for restricted_base in restricted_base_fields:
            # Check if the requested field is exactly a restricted base field
            # or if it's a sub-attribute of a restricted base field (e.g., "performance.feedback.details")
            if field_name == restricted_base or \
               field_name.startswith(restricted_base + ".") or \
               field_name.startswith(restricted_base + "["):
                raise ValueError(
                    f"Access to restricted field '{field_name}' (derived from '{restricted_base}') is not allowed."
                )

    try:
        # Format the string using the employee and (if available) performance objects
        return template_str.format(employee=employee, performance=performance)
    except AttributeError as e:
        # This handles cases like {employee.non_existent_attribute} or {performance.non_existent_attribute}
        # or if performance is None and the template tries to access its attributes.
        raise AttributeError(f"Error formatting template: An attribute was not found. Original error: {e}") from e
    except KeyError as e:
        # This can happen with complex field names like {employee[key]} if 'key' is not suitable.
        raise KeyError(f"Error formatting template: A key was not found. Original error: {e}") from e
