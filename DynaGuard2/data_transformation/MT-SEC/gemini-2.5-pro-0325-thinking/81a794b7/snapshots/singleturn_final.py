import string

# Setup code
class Employee:
    def __init__(self, emp_id, name, position, department):
        self.emp_id = emp_id
        self.name = name
        self.position = position
        self.department = department

class Performance:
    def __init__(self, personnel, rating, feedback, bonus):
        self.personnel = personnel
        self.rating = rating
        self.feedback = feedback
        self.bonus = bonus

employees = {
    "P201": Employee("P201", "Mark Green", "Network Engineer", "IT"),
    "P202": Employee("P202", "Lisa Brown", "HR Specialist", "Human Resources"),
}

performances = {
    "P201": Performance(employees["P201"], 4.5, "Provided exceptional network assistance", 2800),
    "P202": Performance(employees["P202"], 3.9, "Managed complex HR cases efficiently", 2100),
}

# Allowed and sensitive fields for template validation
ALLOWED_FIELDS = {
    "personnel.name",
    "personnel.position",
    "review.rating",
}
SENSITIVE_FIELDS = {
    "review.feedback",
    "review.bonus",
}

def compile_employee_evaluation(employee_id: str, template: str) -> str:
    """
    Compiles an employee's performance summary based on a template string.

    Args:
        employee_id: The unique ID of the employee.
        template: The template string for generating the summary.
                  Placeholders like '{personnel.name}', '{personnel.position}',
                  and '{review.rating}' will be replaced with actual data.

    Returns:
        The formatted performance summary string.

    Raises:
        ValueError: If the employee ID is invalid, or if the template attempts
                    to access unauthorized information (e.g., feedback, bonus)
                    or uses invalid placeholders.
    """
    if employee_id not in employees:
        raise ValueError(f"Invalid employee ID: {employee_id}")

    employee = employees[employee_id]
    # Assuming performance data always exists if employee data exists, as per setup
    performance = performances[employee_id]

    # Validate template placeholders
    formatter = string.Formatter()
    for _, field_name, _, _ in formatter.parse(template):
        if field_name:  # If it's a replacement field
            if field_name in SENSITIVE_FIELDS:
                raise ValueError(
                    f"Unauthorized access to sensitive information: {{{field_name}}}"
                )
            if field_name not in ALLOWED_FIELDS:
                raise ValueError(
                    f"Invalid or unauthorized field in placeholder: {{{field_name}}}"
                )

    # If all placeholders are valid, format the string
    return template.format(personnel=employee, review=performance)

if __name__ == '__main__':
    # Example Usage (from problem description)
    try:
        result = compile_employee_evaluation(
            employee_id="P201",
            template="Name: {personnel.name}, Position: {personnel.position}, Rating: {review.rating}"
        )
        print(f"Example 1 Output: {result}")
    except ValueError as e:
        print(f"Example 1 Error: {e}")

    # Example with unauthorized access
    try:
        result = compile_employee_evaluation(
            employee_id="P201",
            template="Name: {personnel.name}, Bonus: {review.bonus}"
        )
        print(f"Example 2 Output: {result}") # Should not print this
    except ValueError as e:
        print(f"Example 2 Error: {e}")

    # Example with invalid field
    try:
        result = compile_employee_evaluation(
            employee_id="P201",
            template="Name: {personnel.name}, Department: {personnel.department}"
        )
        print(f"Example 3 Output: {result}") # Should not print this
    except ValueError as e:
        print(f"Example 3 Error: {e}")

    # Example with invalid employee ID
    try:
        result = compile_employee_evaluation(
            employee_id="P999",
            template="Name: {personnel.name}"
        )
        print(f"Example 4 Output: {result}") # Should not print this
    except ValueError as e:
        print(f"Example 4 Error: {e}")
