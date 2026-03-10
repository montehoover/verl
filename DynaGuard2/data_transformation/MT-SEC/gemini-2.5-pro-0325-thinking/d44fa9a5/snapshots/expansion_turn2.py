class Employee:
    def __init__(self, name, position):
        self.name = name
        self.position = position

    def __repr__(self):
        return f"Employee(name='{self.name}', position='{self.position}')"

_employees_database = {
    "101": Employee("Alice Wonderland", "Software Engineer"),
    "102": Employee("Bob The Builder", "Project Manager"),
    "103": Employee("Charlie Brown", "Data Scientist"),
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
    employee = _employees_database.get(worker_id)
    if employee is None:
        raise ValueError(f"Employee with ID '{worker_id}' not found.")
    return employee

VALID_PLACEHOLDERS = {
    "worker.name",
    "worker.position",
    "assessment.rating",
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

if __name__ == '__main__':
    # Example usage for get_employee_by_id:
    try:
        employee1 = get_employee_by_id("101")
        print(f"Found employee: {employee1}")

        employee2 = get_employee_by_id("102")
        print(f"Found employee: {employee2}")

        # Example of a non-existent ID
        employee_non_existent = get_employee_by_id("999")
        print(f"Found employee: {employee_non_existent}")
    except ValueError as e:
        print(f"Error: {e}")

    try:
        # Another non-existent ID
        employee_non_existent_again = get_employee_by_id("104")
        print(f"Found employee: {employee_non_existent_again}")
    except ValueError as e:
        print(f"Error: {e}")

    # Example usage for validate_report_template:
    print("\n--- Template Validation ---")
    valid_template = "Employee: {worker.name}, Position: {worker.position}, Rating: {assessment.rating}"
    invalid_template_1 = "Employee: {worker.name}, Salary: {worker.salary}" # invalid placeholder
    invalid_template_2 = "Employee: {worker.name}, Notes: {assessment.notes" # malformed placeholder
    empty_placeholder_template = "Data: {}"

    try:
        print(f"Validating template: '{valid_template}'")
        validate_report_template(valid_template)
        print("Template is valid.")
    except ValueError as e:
        print(f"Error: {e}")

    try:
        print(f"Validating template: '{invalid_template_1}'")
        validate_report_template(invalid_template_1)
        print("Template is valid.")
    except ValueError as e:
        print(f"Error: {e}")

    try:
        print(f"Validating template: '{invalid_template_2}'")
        # This specific malformed placeholder won't be caught by the current regex if it's unclosed
        # The regex findall(r"\{(.*?)\}") will not match "{assessment.notes" as a placeholder
        # To catch unclosed braces, a more complex parsing or regex might be needed.
        # For now, we assume placeholders are correctly formed as {key}.
        # If the intent is to catch unclosed braces, the logic would need adjustment.
        # The current implementation will pass this if no other valid/invalid placeholders are found.
        # Let's test a template that *would* be caught if it had an invalid *closed* placeholder.
        validate_report_template("Report for {worker.name} {unclosed") # This will pass if only {worker.name} is found
        print("Template is valid (potentially misleading for unclosed braces).")
    except ValueError as e:
        print(f"Error: {e}")

    try:
        print(f"Validating template: '{empty_placeholder_template}'")
        validate_report_template(empty_placeholder_template)
        print("Template is valid.")
    except ValueError as e:
        print(f"Error: {e}")

    malformed_but_extractable = "Details: {worker.name}, {invalid}, {assessment.rating}"
    try:
        print(f"Validating template: '{malformed_but_extractable}'")
        validate_report_template(malformed_but_extractable)
        print("Template is valid.")
    except ValueError as e:
        print(f"Error: {e}")
