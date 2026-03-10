class Employee:
    def __init__(self, emp_id: str, name: str, position: str, department: str):
        self.emp_id = emp_id
        self.name = name
        self.position = position
        self.department = department

    def __repr__(self):
        return f"Employee(emp_id='{self.emp_id}', name='{self.name}', position='{self.position}', department='{self.department}')"

class Performance:
    def __init__(self, worker: Employee, rating: float, feedback: str, bonus: int):
        self.worker = worker
        self.rating = rating
        self.feedback = feedback
        self.bonus = bonus

    def __repr__(self):
        return f"Performance(worker_id='{self.worker.emp_id}', rating={self.rating}, feedback='{self.feedback}', bonus={self.bonus})"

employees = {
    "W101": Employee("W101", "Alice Johnson", "Project Manager", "Operations"),
    "W102": Employee("W102", "Bob Lee", "Software Developer", "IT"),
}

performances = {
    "W101": Performance(employees["W101"], 4.8, "Led successful project launches", 3000),
    "W102": Performance(employees["W102"], 4.0, "Developed key modules on time", 2500),
}

def get_employee_by_id(worker_id: str) -> Employee:
    """
    Retrieves an employee object by their worker ID.

    Args:
        worker_id: The ID of the worker to retrieve.

    Returns:
        The Employee object corresponding to the worker ID.

    Raises:
        ValueError: If the worker ID does not exist in the database.
    """
    employee = employees.get(worker_id) # Use the new 'employees' dictionary
    if employee is None:
        raise ValueError(f"Employee with ID '{worker_id}' not found.")
    return employee

VALID_PLACEHOLDERS = {
    "employee.emp_id",
    "employee.name",
    "employee.position",
    "employee.department",
    "performance.rating",
    "performance.feedback",
    "performance.bonus",
}

def validate_report_template(template_string: str) -> bool:
    """
    Validates a report template string for correct placeholders.

    Args:
        template_string: The template string to validate.

    Returns:
        True if all placeholders are valid.

    Raises:
        ValueError: If an invalid or malformed placeholder is found.
    """
    import re
    placeholders = re.findall(r"\{(.*?)\}", template_string)
    for placeholder in placeholders:
        if not placeholder: # Handles cases like {}
            raise ValueError("Empty placeholder '{}' found in template.")
        if placeholder not in VALID_PLACEHOLDERS:
            raise ValueError(f"Invalid placeholder '{{{placeholder}}}' found in template.")
    return True

def create_employee_report(worker_id: str, template: str) -> str:
    """
    Generates a formatted performance summary string for an employee.

    Args:
        worker_id: The ID of the worker.
        template: The template string with placeholders.

    Returns:
        The formatted summary string.

    Raises:
        ValueError: If the worker ID is invalid, performance data is missing,
                    or the template contains unauthorized/incorrect placeholders.
    """
    # Validate template first. This will raise ValueError on issues.
    validate_report_template(template)

    # Get employee details
    employee = get_employee_by_id(worker_id) # Raises ValueError if not found

    # Get performance details
    performance_data = performances.get(worker_id)
    if performance_data is None:
        # This case implies an inconsistency if employees and performances are meant to be synced
        # or if a worker_id is valid for an employee but they have no performance record.
        raise ValueError(f"Performance data not found for worker ID '{worker_id}'.")

    report_string = template

    # Replace employee placeholders
    report_string = report_string.replace("{employee.emp_id}", str(employee.emp_id))
    report_string = report_string.replace("{employee.name}", str(employee.name))
    report_string = report_string.replace("{employee.position}", str(employee.position))
    report_string = report_string.replace("{employee.department}", str(employee.department))

    # Replace performance placeholders
    report_string = report_string.replace("{performance.rating}", str(performance_data.rating))
    report_string = report_string.replace("{performance.feedback}", str(performance_data.feedback))
    report_string = report_string.replace("{performance.bonus}", str(performance_data.bonus))
    
    return report_string

if __name__ == '__main__':
    # Example Usage for create_employee_report

    print("--- Employee Report Generation ---")

    # Valid scenario
    template1 = "Employee Report:\nID: {employee.emp_id}\nName: {employee.name}\nPosition: {employee.position}\nDepartment: {employee.department}\nPerformance Rating: {performance.rating}\nFeedback: {performance.feedback}\nBonus: ${performance.bonus}"
    try:
        report1 = create_employee_report("W101", template1)
        print("\nReport for W101 (Valid Template):")
        print(report1)
    except ValueError as e:
        print(f"Error generating report for W101: {e}")

    # Another valid scenario
    try:
        report2 = create_employee_report("W102", template1)
        print("\nReport for W102 (Valid Template):")
        print(report2)
    except ValueError as e:
        print(f"Error generating report for W102: {e}")

    # Invalid worker ID
    try:
        print("\nAttempting report for W999 (Invalid Worker ID):")
        report_invalid_worker = create_employee_report("W999", template1)
        print(report_invalid_worker)
    except ValueError as e:
        print(f"Error: {e}")

    # Invalid template (unauthorized placeholder)
    template_invalid_placeholder = "Name: {employee.name}, Salary: {employee.salary}" # {employee.salary} is not valid
    try:
        print(f"\nAttempting report for W101 with invalid template: '{template_invalid_placeholder}'")
        report_invalid_template = create_employee_report("W101", template_invalid_placeholder)
        print(report_invalid_template)
    except ValueError as e:
        print(f"Error: {e}")

    # Invalid template (empty placeholder)
    template_empty_placeholder = "Name: {employee.name}, Info: {}"
    try:
        print(f"\nAttempting report for W101 with empty placeholder template: '{template_empty_placeholder}'")
        report_empty_placeholder = create_employee_report("W101", template_empty_placeholder)
        print(report_empty_placeholder)
    except ValueError as e:
        print(f"Error: {e}")
    
    # Example: Employee exists but no performance data (requires manual data setup to test)
    # Add a temporary employee without performance data to test this specific case:
    employees["W103"] = Employee("W103", "Carol Danvers", "Chief of Security", "Security")
    try:
        print("\nAttempting report for W103 (No Performance Data):")
        # This template is valid, but performance data for W103 is missing.
        report_no_perf = create_employee_report("W103", "Name: {employee.name}, Rating: {performance.rating}")
        print(report_no_perf)
    except ValueError as e:
        print(f"Error: {e}")
    finally:
        if "W103" in employees:
            del employees["W103"] # Clean up temporary data

    print("\n--- Sanity check: get_employee_by_id (adapted) ---")
    try:
        emp = get_employee_by_id("W101")
        print(f"Found employee W101: {emp}")
        emp_non_existent = get_employee_by_id("W300") # Should raise ValueError
        print(emp_non_existent) 
    except ValueError as e:
        print(f"Error with get_employee_by_id: {e}")

    print("\n--- Sanity check: validate_report_template (with new placeholders) ---")
    valid_tpl = "Name: {employee.name}, Rating: {performance.rating}"
    invalid_tpl_field = "Name: {employee.name}, Secret: {employee.secret_info}"
    try:
        validate_report_template(valid_tpl)
        print(f"Template '{valid_tpl}' is valid.")
    except ValueError as e:
        print(f"Error validating '{valid_tpl}': {e}")
    
    try:
        validate_report_template(invalid_tpl_field) # Should raise error
        print(f"Template '{invalid_tpl_field}' is valid.") # Should not reach here
    except ValueError as e:
        print(f"Error validating '{invalid_tpl_field}': {e}")
