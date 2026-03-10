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

def generate_employee_summary(employee_id: str, template: str) -> str:
    """
    Generates an employee summary string based on performance data and a template.

    Args:
        employee_id: The ID of the employee.
        template: The summary template string with placeholders.
                  Placeholders should be in the format {object.attribute},
                  e.g., {employee.name} or {performance.rating}.

    Returns:
        The generated summary string.

    Raises:
        ValueError: If any error occurs, including employee not found,
                    invalid placeholders, or attempted access to sensitive
                    information (feedback or bonus details).
    """
    if not isinstance(employee_id, str):
        raise ValueError("Employee ID must be a string.")
    if not isinstance(template, str):
        raise ValueError("Template must be a string.")

    if employee_id not in employees:
        raise ValueError(f"Employee with ID '{employee_id}' not found.")
    if employee_id not in performances:
        # Assuming performance data must exist for a summary to be generated.
        raise ValueError(f"Performance record for employee ID '{employee_id}' not found.")

    employee_obj = employees[employee_id]
    performance_obj = performances[employee_id]

    def replacer(match):
        object_name = match.group(1)
        attribute_name = match.group(2)

        if object_name == "employee":
            try:
                # Ensure we don't allow access to private/protected attributes if any were added later
                if attribute_name.startswith('_'):
                    raise AttributeError(f"Access to private/protected attribute '{attribute_name}' is not allowed.")
                value = getattr(employee_obj, attribute_name)
                return str(value)
            except AttributeError:
                raise ValueError(f"Employee has no attribute '{attribute_name}'. Invalid placeholder: {{employee.{attribute_name}}}")
        elif object_name == "performance":
            if attribute_name in ["feedback", "bonus"]:
                raise ValueError(f"Access to sensitive performance data ('{attribute_name}') is not allowed. Invalid placeholder: {{performance.{attribute_name}}}")
            
            if attribute_name == "employee":
                # {performance.employee} would refer to the Employee object.
                # Stringifying it isn't useful, and employee data should be accessed via {employee.attribute}.
                raise ValueError("Placeholder '{performance.employee}' is not supported. Access employee attributes directly via '{employee.attribute_name}'.")
            
            try:
                # Ensure we don't allow access to private/protected attributes
                if attribute_name.startswith('_'):
                    raise AttributeError(f"Access to private/protected attribute '{attribute_name}' is not allowed.")
                value = getattr(performance_obj, attribute_name)
                return str(value)
            except AttributeError:
                raise ValueError(f"Performance record has no attribute '{attribute_name}'. Invalid placeholder: {{performance.{attribute_name}}}")
        else:
            raise ValueError(f"Unknown object type '{object_name}' in placeholder '{{{object_name}.{attribute_name}}}'")

    # The regex r'{(\w+)\.(\w+)}' matches placeholders like {object.attribute}
    # \w+ matches one or more word characters (letters, numbers, underscore)
    summary = re.sub(r'{(\w+)\.(\w+)}', replacer, template)
    
    return summary
