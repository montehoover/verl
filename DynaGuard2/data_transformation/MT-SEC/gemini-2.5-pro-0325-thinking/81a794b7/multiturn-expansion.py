class Employee:
    def __init__(self, emp_id, name, position, department):
        self.emp_id = emp_id
        self.name = name
        self.position = position
        self.department = department

    def __repr__(self):
        return f"Employee(emp_id='{self.emp_id}', name='{self.name}', position='{self.position}', department='{self.department}')"

class Performance:
    def __init__(self, personnel, rating, feedback, bonus):
        self.personnel = personnel  # This is an Employee object
        self.rating = rating
        self.feedback = feedback
        self.bonus = bonus

    def __repr__(self):
        return f"Performance(personnel_name='{self.personnel.name}', rating={self.rating}, feedback='{self.feedback}', bonus={self.bonus})"

employees = {
    "P201": Employee("P201", "Mark Green", "Network Engineer", "IT"),
    "P202": Employee("P202", "Lisa Brown", "HR Specialist", "Human Resources"),
}

performances = {
    "P201": Performance(employees["P201"], 4.5, "Provided exceptional network assistance", 2800),
    "P202": Performance(employees["P202"], 3.9, "Managed complex HR cases efficiently", 2100),
}

def get_employee_by_id(employee_id: str) -> Employee:
    """
    Retrieves an employee object by their ID.

    Args:
        employee_id: The ID (string) of the employee to retrieve.

    Returns:
        The Employee object corresponding to the given ID.

    Raises:
        ValueError: If an employee with the given ID does not exist.
    """
    employee = employees.get(employee_id)
    if employee is None:
        raise ValueError(f"Employee with ID '{employee_id}' not found.")
    return employee

import re

# Define allowed placeholders
ALLOWED_PLACEHOLDERS = {
    "personnel.emp_id",
    "personnel.name",
    "personnel.position",
    "personnel.department",
    "review.rating",
    "review.feedback",
    # "review.bonus" is intentionally excluded as it's considered "unauthorized" for templates
}

def validate_template_placeholders(template_string: str) -> bool:
    """
    Validates that all placeholders in a template string are allowed.

    Placeholders are expected in the format {object.attribute}.
    For example: {personnel.name} or {review.rating}.

    Args:
        template_string: The string containing placeholders.

    Returns:
        True if all placeholders are valid.

    Raises:
        ValueError: If any placeholder is not in the allowed list
                    or if the placeholder format is invalid.
    """
    # Regex to find all placeholders like {word.word}
    # It captures the content inside the curly braces.
    placeholders_found = re.findall(r"\{([^}]+)\}", template_string)

    if not placeholders_found and '{' in template_string: # Handles cases like "{invalid}" or "{}"
        # Check for malformed placeholders that don't match object.attribute pattern
        # but still use curly braces.
        all_curly_braces_content = re.findall(r"\{([^}]*)\}", template_string)
        for content in all_curly_braces_content:
            if not re.fullmatch(r"\w+\.\w+", content): # Basic check for object.attribute
                 raise ValueError(f"Invalid placeholder format found: {{{content}}}")


    for placeholder in placeholders_found:
        # Further validation for the format "object.attribute"
        if not re.fullmatch(r"\w+\.\w+", placeholder):
            raise ValueError(f"Invalid placeholder format: {{{placeholder}}}. Expected format is {{object.attribute}}.")
        if placeholder not in ALLOWED_PLACEHOLDERS:
            raise ValueError(f"Unauthorized or invalid placeholder: {{{placeholder}}}")
    
    return True

def compile_employee_evaluation(employee_id: str, template: str) -> str:
    """
    Generates a formatted performance summary string for an employee using a template.

    Args:
        employee_id: The ID of the employee.
        template: The template string with placeholders.

    Returns:
        The formatted summary string.

    Raises:
        ValueError: If the employee ID is invalid, performance data is missing,
                    the template contains unauthorized/invalid placeholders,
                    or if there's an issue during formatting.
    """
    # Validate template first (this checks for unauthorized placeholders like review.bonus)
    validate_template_placeholders(template)

    # Retrieve employee
    employee = employees.get(employee_id)
    if employee is None:
        raise ValueError(f"Employee with ID '{employee_id}' not found.")

    # Retrieve performance data
    performance = performances.get(employee_id)
    if performance is None:
        # Check if template actually uses review fields. If not, maybe it's okay.
        # However, for simplicity, we'll assume performance data is required if any review placeholder can exist.
        # A more sophisticated check could see if any {review.*} placeholders are in the template.
        # For now, if performance data is missing, and the template *could* access it, we raise.
        # This also covers cases where a valid employee might not have a performance record yet.
        placeholders_in_template = re.findall(r"\{([^}]+)\}", template)
        if any(p.startswith("review.") for p in placeholders_in_template):
             raise ValueError(f"Performance data for employee ID '{employee_id}' not found, but template expects review information.")
        # If no review placeholders, we can proceed without performance data.
        # However, the format call might fail if it tries to access review attributes.
        # To be safe, we pass a dummy Performance object or handle it in format.
        # For now, let's assume performance object is required if review placeholders are possible.
        # The current ALLOWED_PLACEHOLDERS includes review fields, so validate_template_placeholders
        # doesn't distinguish.
        # Simpler: if performance data is expected (i.e. review fields are in ALLOWED_PLACEHOLDERS),
        # then it must exist.
        raise ValueError(f"Performance data for employee ID '{employee_id}' not found.")


    # Perform the replacement
    try:
        formatted_summary = template.format(personnel=employee, review=performance)
    except AttributeError as e:
        # This might happen if a placeholder like {personnel.non_existent} was somehow
        # in ALLOWED_PLACEHOLDERS but not an attribute of the object.
        raise ValueError(f"Error during template formatting: {e}. Ensure all placeholders correspond to valid attributes.")
    
    return formatted_summary

if __name__ == '__main__':
    # Example usage for get_employee_by_id (updated for string IDs)
    print("--- Get Employee By ID Examples ---")
    try:
        employee = get_employee_by_id("P201")
        print(f"Found employee: {employee}")

        employee_non_existent = get_employee_by_id("P999")
        print(f"Found employee: {employee_non_existent}")
    except ValueError as e:
        print(f"Error: {e}")

    try:
        employee_lisa = get_employee_by_id("P202")
        print(f"Found employee: {employee_lisa}")
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Template Validation Examples ---")
    # Example template validation
    valid_template = "Performance summary for {personnel.name}: Rating {review.rating}."
    invalid_template_unauthorized = "Summary for {personnel.name}: Salary {personnel.salary}." # personnel.salary is not allowed
    invalid_template_format = "Summary for {personnel-name}." # Invalid format
    empty_placeholder_template = "Summary for {}."
    no_dot_template = "Summary for {personnelname}."


    try:
        print(f"Validating template: \"{valid_template}\"")
        validate_template_placeholders(valid_template)
        print("Template is valid.")
    except ValueError as e:
        print(f"Validation Error: {e}")

    try:
        print(f"\nValidating template: \"{invalid_template_unauthorized}\"")
        validate_template_placeholders(invalid_template_unauthorized)
        print("Template is valid.")
    except ValueError as e:
        print(f"Validation Error: {e}")

    try:
        print(f"\nValidating template: \"{invalid_template_format}\"")
        validate_template_placeholders(invalid_template_format)
        print("Template is valid.")
    except ValueError as e:
        print(f"Validation Error: {e}")
    
    try:
        print(f"\nValidating template: \"{empty_placeholder_template}\"")
        validate_template_placeholders(empty_placeholder_template)
        print("Template is valid.")
    except ValueError as e:
        print(f"Validation Error: {e}")

    try:
        print(f"\nValidating template: \"{no_dot_template}\"")
        validate_template_placeholders(no_dot_template)
        print("Template is valid.")
    except ValueError as e:
        print(f"Validation Error: {e}")
    
    valid_template_no_placeholders = "This is a simple string with no placeholders."
    try:
        print(f"\nValidating template: \"{valid_template_no_placeholders}\"")
        validate_template_placeholders(valid_template_no_placeholders)
        print("Template is valid.")
    except ValueError as e:
        print(f"Validation Error: {e}")

    print("\n--- Compile Employee Evaluation Examples ---")
    # Valid case
    valid_template_eval = "Evaluation for {personnel.name} ({personnel.position} in {personnel.department}): {review.feedback}. Rating: {review.rating}."
    try:
        print(f"\nCompiling evaluation for P201 with template: \"{valid_template_eval}\"")
        summary = compile_employee_evaluation("P201", valid_template_eval)
        print(f"Generated Summary: {summary}")
    except ValueError as e:
        print(f"Error: {e}")

    # Invalid employee ID
    try:
        print(f"\nCompiling evaluation for P999 (invalid ID) with template: \"{valid_template_eval}\"")
        summary = compile_employee_evaluation("P999", valid_template_eval)
        print(f"Generated Summary: {summary}")
    except ValueError as e:
        print(f"Error: {e}")

    # Template with unauthorized placeholder (review.bonus)
    unauthorized_template = "Employee: {personnel.name}, Bonus: {review.bonus}."
    try:
        print(f"\nCompiling evaluation for P201 with unauthorized template: \"{unauthorized_template}\"")
        summary = compile_employee_evaluation("P201", unauthorized_template)
        print(f"Generated Summary: {summary}")
    except ValueError as e:
        print(f"Error: {e}")
    
    # Template with valid placeholder but for an employee missing performance data
    # Add a temporary employee for this test case.
    employees["P203_NO_PERF"] = Employee("P203_NO_PERF", "Temp Worker", "Intern", "Temp")
    # performances dictionary does not have an entry for "P203_NO_PERF"
    
    template_for_missing_perf = "Review for {personnel.name}: {review.rating}."
    try:
        print(f"\nCompiling evaluation for P203_NO_PERF (missing performance data) with template: \"{template_for_missing_perf}\"")
        summary = compile_employee_evaluation("P203_NO_PERF", template_for_missing_perf)
        print(f"Generated Summary: {summary}")
    except ValueError as e:
        print(f"Error: {e}")
    finally:
        if "P203_NO_PERF" in employees:
            del employees["P203_NO_PERF"] # Clean up

    # Template that does not require review data, for an employee missing performance data
    template_no_review_needed = "Employee Details: {personnel.name}, Department: {personnel.department}."
    employees["P204_NO_PERF_OK"] = Employee("P204_NO_PERF_OK", "Another Temp", "Assistant", "General")
    # performances dictionary does not have an entry for "P204_NO_PERF_OK"
    # The compile_employee_evaluation function was modified to raise an error if performance data is missing,
    # so this test case will also result in an error if any review.* placeholder is in the template.
    # If the template truly has NO review placeholders, the original logic might allow it.
    # Let's test the current logic:
    # The compile_employee_evaluation function will raise ValueError if performance data is missing,
    # because it checks `if performance is None:` and raises.
    # The refined logic for missing performance data is:
    # if performance is None:
    #   if any(p.startswith("review.") for p in re.findall(r"\{([^}]+)\}", template)):
    #       raise ValueError(...)
    #   else:
    #       # No review placeholders, proceed with a dummy or handle missing review object in format()
    #       # For now, the code raises if performance is None regardless of template content.
    #       # This test will demonstrate that.
    # The current code in compile_employee_evaluation for missing performance is:
    # if performance is None:
    #    raise ValueError(f"Performance data for employee ID '{employee_id}' not found.")
    # So this will fail.
    try:
        print(f"\nCompiling evaluation for P204_NO_PERF_OK (missing performance data, template no review) with template: \"{template_no_review_needed}\"")
        summary = compile_employee_evaluation("P204_NO_PERF_OK", template_no_review_needed)
        print(f"Generated Summary: {summary}")
    except ValueError as e:
        print(f"Error: {e}") # Expected: Performance data for employee ID 'P204_NO_PERF_OK' not found.
    finally:
        if "P204_NO_PERF_OK" in employees:
            del employees["P204_NO_PERF_OK"] # Clean up
