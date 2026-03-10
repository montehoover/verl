import string


class Performance:
    def __init__(self, rating, feedback=None, bonus=None):
        self.rating = rating
        self.feedback = feedback
        self.bonus = bonus


class Employee:
    def __init__(self, emp_id, name, position, department, performance):
        self.emp_id = emp_id
        self.name = name
        self.position = position
        self.department = department
        self.performance = performance


employees = {
    "E101": Employee(
        "E101",
        "John Doe",
        "Senior Software Engineer",
        "R&D",
        Performance("Exceeds Expectations", feedback="Strong leadership", bonus=7500),
    ),
    "E102": Employee(
        "E102",
        "Jane Smith",
        "Business Analyst",
        "Operations",
        Performance("Meets Expectations", feedback="Reliable and consistent", bonus=5000),
    ),
}


def generate_employee_summary(employee_id: str, template: str) -> str:
    """
    Return a formatted summary string generated from the provided template.

    The template may include placeholders such as:
      - '{employee.name}'
      - '{employee.position}'
      - '{performance.rating}'

    Access to sensitive information such as feedback or bonus details is not allowed.

    Args:
        employee_id: The employee's ID.
        template: A string containing placeholders referencing employee or performance attributes.

    Returns:
        A formatted summary string based on the template.

    Raises:
        ValueError: If the employee ID does not exist or if access to sensitive information is attempted.
    """
    emp = employees.get(employee_id)
    if emp is None:
        raise ValueError(f"Employee ID '{employee_id}' does not exist.")

    # Check for sensitive fields in the template placeholders (feedback, bonus)
    formatter = string.Formatter()
    for _, field_name, _, _ in formatter.parse(template):
        if not field_name:
            continue
        # Normalize the field path and check tokens
        normalized = field_name.replace("[", ".").replace("]", "")
        tokens = [t for t in normalized.split(".") if t]
        if any(tok in {"feedback", "bonus"} for tok in tokens):
            raise ValueError("Access to sensitive information is not permitted.")

    try:
        return template.format(employee=emp, performance=emp.performance)
    except Exception as exc:
        # Any formatting error due to invalid placeholders
        raise ValueError("Invalid template.") from exc
