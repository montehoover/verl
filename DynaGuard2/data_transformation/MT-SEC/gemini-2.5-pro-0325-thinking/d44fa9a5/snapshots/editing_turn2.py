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

def get_employee_report(worker_id, format_template):
    """
    Formats employee details based on a template.

    Args:
        worker_id (str): The ID of the employee.
        format_template (str): A string template with placeholders like
                               {worker_id}, {worker_name}, {worker_position},
                               {worker_department}.

    Returns:
        str: The formatted string with employee details.

    Raises:
        ValueError: If the worker_id is invalid or not found.
                    If unauthorized information (e.g. feedback, bonus)
                    is attempted to be accessed via the template (though current
                    implementation only substitutes basic details).
    """
    if worker_id not in employees:
        raise ValueError(f"Invalid worker ID: {worker_id}")
    
    employee = employees[worker_id]
    
    # The problem states: "Raise a ValueError if unauthorized information like feedback or bonus is accessed"
    # This function only makes basic details available for formatting.
    # If the template tried to access, e.g., performance data, that would be an issue,
    # but here we explicitly define what can be substituted.

    # Available placeholders and their corresponding values
    substitutions = {
        "{worker_id}": employee.emp_id,
        "{worker_name}": employee.name,
        "{worker_position}": employee.position,
        "{worker_department}": employee.department,
    }

    report = format_template
    for placeholder, value in substitutions.items():
        report = report.replace(placeholder, str(value))
    
    # Check if template tries to access unauthorized fields (e.g. bonus, feedback)
    # This is a simple check; more robust parsing might be needed for complex templates.
    if "{bonus}" in report or "{feedback}" in report or "{rating}" in report: # Check against original template to avoid false positives after substitution
        if "{bonus}" in format_template or \
           "{feedback}" in format_template or \
           "{rating}" in format_template:
            raise ValueError("Access to unauthorized information (e.g., bonus, feedback, rating) in template is not allowed.")

    return report

if __name__ == '__main__':
    # Example usage:
    report_format_v1 = "Employee Report:\nID: {worker_id}\nName: {worker_name}\nPosition: {worker_position}\nDepartment: {worker_department}"
    report_format_v2 = "Worker: {worker_name} ({worker_id}), Role: {worker_position}, Team: {worker_department}. Unknown: {unknown_placeholder}"

    print("Generating report for W101 (Format 1):")
    try:
        report1 = get_employee_report("W101", report_format_v1)
        print(report1)
    except ValueError as e:
        print(e)

    print("\nGenerating report for W102 (Format 2):")
    try:
        report2 = get_employee_report("W102", report_format_v2)
        print(report2)
    except ValueError as e:
        print(e)

    print("\nGenerating report for W103 (invalid worker):")
    try:
        report3 = get_employee_report("W103", report_format_v1)
        print(report3)
    except ValueError as e:
        print(e)

    print("\nAttempting to generate report with unauthorized fields in template:")
    report_format_unauthorized = "Employee: {worker_name}, Bonus: {bonus}" # Placeholder for bonus
    try:
        report_unauth = get_employee_report("W101", report_format_unauthorized)
        print(report_unauth)
    except ValueError as e:
        print(e)
