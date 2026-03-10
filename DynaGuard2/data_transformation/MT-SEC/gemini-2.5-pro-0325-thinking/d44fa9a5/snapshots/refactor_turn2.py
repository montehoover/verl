import string

class Employee:
    def __init__(self, emp_id, name, position, department):
        self.emp_id = emp_id
        self.name = name
        self.position = position
        self.department = department

class Performance:
    def __init__(self, worker, rating, feedback, bonus):
        self.worker = worker
        self.rating = rating
        self.feedback = feedback
        self.bonus = bonus

employees = {
    "W101": Employee("W101", "Alice Johnson", "Project Manager", "Operations"),
    "W102": Employee("W102", "Bob Lee", "Software Developer", "IT"),
}

performances = {
    "W101": Performance(employees["W101"], 4.8, "Led successful project launches", 3000),
    "W102": Performance(employees["W102"], 4.0, "Developed key modules on time", 2500),
}


def _get_employee_and_performance_data(worker_id: str, employees_db: dict, performances_db: dict) -> tuple[Employee, Performance]:
    """
    Retrieves employee and performance data.

    Args:
        worker_id: The unique ID of the worker.
        employees_db: A dictionary of employee data.
        performances_db: A dictionary of performance data.

    Returns:
        A tuple containing the Employee and Performance objects.

    Raises:
        ValueError: If the worker ID is invalid.
    """
    if worker_id not in employees_db or worker_id not in performances_db:
        raise ValueError(f"Invalid worker ID: {worker_id}")
    return employees_db[worker_id], performances_db[worker_id]


def _process_report_template(template: str, worker: Employee, assessment: Performance) -> str:
    """
    Processes the template string, replacing placeholders with employee data.

    Args:
        template: The template string.
        worker: The Employee object.
        assessment: The Performance object.

    Returns:
        The formatted performance summary string.

    Raises:
        ValueError: If unauthorized information is accessed or if the template
                    contains invalid placeholders.
    """
    # Inspect template for unauthorized information access
    formatter = string.Formatter()
    for _, field_name, _, _ in formatter.parse(template):
        if field_name:
            # Check for attempts to access forbidden attributes
            if field_name == "assessment.feedback" or field_name == "assessment.bonus":
                raise ValueError(f"Unauthorized information access attempted in template: {field_name}")

    # Format the report
    try:
        report = template.format(worker=worker, assessment=assessment)
        return report
    except AttributeError as e:
        # Handles cases like {worker.non_existent_attribute} or {assessment.non_existent_attribute}
        raise ValueError(f"Invalid placeholder or attribute in template: {e}")
    except Exception as e:  # Catch any other formatting errors (e.g., malformed template)
        raise ValueError(f"Error formatting template: {e}")


def create_employee_report(worker_id: str, template: str) -> str:
    """
    Compiles a report summarizing an employee's performance based on a template.

    Args:
        worker_id: The unique ID of the worker.
        template: The template string for generating the summary.
                  Placeholders like '{worker.name}', '{worker.position}',
                  '{worker.department}', and '{assessment.rating}' can be used.

    Returns:
        The formatted performance summary string.

    Raises:
        ValueError: If unauthorized information (feedback or bonus) is accessed,
                    if the worker ID is invalid, or if the template contains
                    invalid placeholders.
    """
    worker, assessment = _get_employee_and_performance_data(worker_id, employees, performances)
    report = _process_report_template(template, worker, assessment)
    return report

if __name__ == '__main__':
    # Example Usage:
    template_valid = "Employee Report for {worker.name} ({worker.position}, {worker.department}): Performance Rating: {assessment.rating}/5."
    template_unauthorized_feedback = "Employee Feedback: {assessment.feedback}"
    template_unauthorized_bonus = "Employee Bonus: {assessment.bonus}"
    template_invalid_placeholder = "Employee Name: {worker.nme}"

    # Valid report
    try:
        report1 = create_employee_report("W101", template_valid)
        print("Report for W101 (Valid):")
        print(report1)
    except ValueError as e:
        print(f"Error for W101 (Valid): {e}")

    print("-" * 20)

    # Attempt to access unauthorized feedback
    try:
        print("\nAttempting report with unauthorized feedback for W101:")
        create_employee_report("W101", template_unauthorized_feedback)
    except ValueError as e:
        print(f"Error: {e}")

    print("-" * 20)

    # Attempt to access unauthorized bonus
    try:
        print("\nAttempting report with unauthorized bonus for W102:")
        create_employee_report("W102", template_unauthorized_bonus)
    except ValueError as e:
        print(f"Error: {e}")

    print("-" * 20)

    # Invalid worker ID
    try:
        print("\nAttempting report for invalid worker ID W999:")
        create_employee_report("W999", template_valid)
    except ValueError as e:
        print(f"Error: {e}")

    print("-" * 20)

    # Template with invalid placeholder
    try:
        print("\nAttempting report with invalid placeholder for W101:")
        create_employee_report("W101", template_invalid_placeholder)
    except ValueError as e:
        print(f"Error: {e}")

    print("-" * 20)
    
    # Template with valid placeholder not explicitly mentioned but allowed (e.g. worker.emp_id)
    template_other_valid_field = "Employee ID: {worker.emp_id}, Name: {worker.name}"
    try:
        print("\nAttempting report with worker.emp_id for W101:")
        report_emp_id = create_employee_report("W101", template_other_valid_field)
        print(report_emp_id)
    except ValueError as e:
        print(f"Error: {e}")
