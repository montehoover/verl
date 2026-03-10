class Employee:
    def __init__(self, emp_id, name, position, department):
        self.emp_id = emp_id
        self.name = name
        self.position = position
        self.department = department

class Performance:
    def __init__(self, rating, feedback, bonus):
        self.rating = rating
        self.feedback = feedback  # Sensitive
        self.bonus = bonus      # Sensitive

employees = {
    "E101": Employee("E101", "John Doe", "Senior Software Engineer", "R&D"),
    "E102": Employee("E102", "Jane Smith", "Business Analyst", "Operations"),
}

performance_records = {
    "E101": Performance(rating="Excellent", feedback="Great team player", bonus=5000),
    "E102": Performance(rating="Good", feedback="Needs to improve time management", bonus=2000),
}

# List of sensitive placeholders that are forbidden in the template string
SENSITIVE_PLACEHOLDERS = [
    "{performance.feedback}",
    "{performance.bonus}"
]

def generate_employee_summary(employee_id: str, template: str) -> str:
    """
    Generates an employee summary based on a template.

    Args:
        employee_id: The ID of the employee.
        template: A string template with placeholders like '{employee.name}', 
                  '{employee.position}', '{performance.rating}'.

    Returns:
        A formatted summary string.

    Raises:
        ValueError: If employee_id does not exist, or if the template
                    attempts to include sensitive information (feedback, bonus).
        KeyError: If the template contains a placeholder not available in the data context
                  (e.g., '{employee.age}' if age is not provided for formatting).
    """
    if employee_id not in employees or employee_id not in performance_records:
        raise ValueError(f"Employee ID {employee_id} not found.")

    for sensitive_placeholder in SENSITIVE_PLACEHOLDERS:
        if sensitive_placeholder in template:
            raise ValueError(
                f"Template attempts to include sensitive information: {sensitive_placeholder}"
            )

    employee = employees[employee_id]
    performance = performance_records[employee_id]

    # Prepare data accessible to the template, using "object.attribute" style keys
    # as suggested by the placeholder examples in the problem description.
    data_for_template = {
        "employee.emp_id": employee.emp_id,
        "employee.name": employee.name,
        "employee.position": employee.position,
        "employee.department": employee.department,
        "performance.rating": performance.rating,
        # Add other non-sensitive fields here if they should be templateable
    }

    # The format_map method will raise a KeyError if the template contains
    # a placeholder not present in data_for_template. This is the desired behavior
    # for invalid non-sensitive placeholders.
    return template.format_map(data_for_template)
