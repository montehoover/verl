# Define Employee and Performance classes and their data
class Employee:
    def __init__(self, emp_id, name, position, department):
        self.emp_id = emp_id
        self.name = name
        self.position = position
        self.department = department

    def __repr__(self):
        return f"Employee(emp_id='{self.emp_id}', name='{self.name}')"

class Performance:
    def __init__(self, employee, rating, feedback, bonus):
        self.employee = employee  # Employee object
        self.rating = rating
        self.feedback = feedback
        self.bonus = bonus

    def __repr__(self):
        # Truncate feedback for concise representation
        feedback_repr = self.feedback
        if len(feedback_repr) > 20:
            feedback_repr = feedback_repr[:20] + "..."
        return f"Performance(employee='{self.employee.name}', rating={self.rating}, feedback='{feedback_repr}', bonus={self.bonus})"

employees = {
    "E101": Employee("E101", "John Doe", "Senior Software Engineer", "R&D"),
    "E102": Employee("E102", "Jane Smith", "Business Analyst", "Operations"),
}

performances = {
    "E101": Performance(employees["E101"], 4.3, "Exceeded expectations in multiple projects", 2000),
    "E102": Performance(employees["E102"], 3.8, "Consistently met expectations.", 1500),
}

def get_performance_by_id(employee_id: str) -> Performance:
    """
    Retrieves the Performance object for a given employee ID.

    Args:
        employee_id: The ID of the employee.

    Returns:
        The Performance object associated with the employee ID.

    Raises:
        ValueError: If the employee ID is not found in the data.
    """
    if employee_id in performances:
        return performances[employee_id]
    else:
        raise ValueError(f"Employee ID '{employee_id}' not found in performance data.")

# Define allowed placeholders for summary templates
VALID_PLACEHOLDERS = {
    "{employee.name}",
    "{employee.position}",
    "{employee.department}",
    "{employee.emp_id}",
    "{performance.rating}",
    "{performance.feedback}",
    "{performance.bonus}",
}

# It's good practice to import `re` at the top of the file,
# but for this incremental change, we'll add it here.
# If this were a real project, I'd move it to the top.
import re

def check_summary_placeholders(template_string: str) -> bool:
    """
    Checks if a summary template string contains only valid placeholders.

    Args:
        template_string: The template string to check.

    Returns:
        True if all placeholders are valid.

    Raises:
        ValueError: If any invalid or sensitive placeholders are found.
    """
    # Find all placeholders in the template string (e.g., {placeholder_name})
    found_placeholders = set(re.findall(r"({[^}]+})", template_string))

    for placeholder in found_placeholders:
        if placeholder not in VALID_PLACEHOLDERS:
            # For security, avoid echoing the potentially malicious placeholder directly in the error message
            # if it could be very long or contain harmful characters.
            # For this example, simple reporting is fine.
            raise ValueError(f"Invalid or sensitive placeholder found: {placeholder}")
    
    return True

def generate_employee_summary(employee_id: str, template: str) -> str:
    """
    Generates an employee performance summary using a template.

    Args:
        employee_id: The ID of the employee.
        template: The template string for the summary.

    Returns:
        The generated summary string.

    Raises:
        ValueError: If employee ID is not found, or if the template
                    contains sensitive or unknown/disallowed placeholders.
    """
    if employee_id not in employees:
        raise ValueError(f"Employee ID '{employee_id}' not found in employee data.")
    if employee_id not in performances:
        raise ValueError(f"Performance data for employee ID '{employee_id}' not found.")

    employee_obj = employees[employee_id]
    performance_obj = performances[employee_id]

    # Placeholders allowed for substitution in this specific summary type
    allowed_data_for_template = {
        "{employee.name}": employee_obj.name,
        "{employee.position}": employee_obj.position,
        "{employee.department}": employee_obj.department,
        "{performance.rating}": str(performance_obj.rating),
    }

    # Sensitive placeholders that are strictly forbidden in this summary
    sensitive_placeholders = {"{performance.feedback}", "{performance.bonus}"}

    summary = template
    found_placeholders = set(re.findall(r"({[^}]+})", template))

    for ph in found_placeholders:
        if ph in sensitive_placeholders:
            raise ValueError(f"Template contains sensitive placeholder: {ph}")
        if ph not in allowed_data_for_template:
            raise ValueError(f"Template contains unknown or disallowed placeholder for summary: {ph}")
        
        # Substitute valid, non-sensitive placeholders
        summary = summary.replace(ph, allowed_data_for_template[ph])
            
    return summary

if __name__ == '__main__':
    print("--- Testing get_performance_by_id ---")
    try:
        perf_e101 = get_performance_by_id("E101")
        print(f"Performance for E101: {perf_e101}")
        perf_e102 = get_performance_by_id("E102")
        print(f"Performance for E102: {perf_e102}")
    except ValueError as e:
        print(f"Error: {e}")

    try:
        print("\nAttempting to get performance for non-existent ID E103:")
        # E103 is not in the new `performances` data
        get_performance_by_id("E103")
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Testing check_summary_placeholders ---")
    # Test cases for check_summary_placeholders with updated VALID_PLACEHOLDERS
    test_templates_check = {
        "Valid simple": "Name: {employee.name}, Rating: {performance.rating}",
        "Valid complex": "EmpID: {employee.emp_id}, Dept: {employee.department}, Feedback: {performance.feedback}, Bonus: {performance.bonus}",
        "Invalid placeholder": "Salary: {employee.salary}", # Not in VALID_PLACEHOLDERS
        "Partially valid": "Name: {employee.name}, Invalid: {foo.bar}", # {foo.bar} is invalid
        "Empty": "No placeholders here."
    }
    for name, template_str in test_templates_check.items():
        try:
            if check_summary_placeholders(template_str):
                print(f"'{name}' template is valid by check_summary_placeholders: \"{template_str}\"")
        except ValueError as e:
            print(f"Error for '{name}' template \"{template_str}\" by check_summary_placeholders: {e}")
    
    print("\n--- Testing generate_employee_summary ---")
    # Test cases for generate_employee_summary
    summary_template_valid = "Summary for {employee.name} ({employee.position}): Rating of {performance.rating} in {employee.department}."
    summary_template_sensitive_feedback = "Summary for {employee.name}: {performance.feedback}."
    summary_template_sensitive_bonus = "Summary for {employee.name}: Bonus {performance.bonus}."
    summary_template_unknown = "Summary for {employee.name}: Unknown data {unknown.placeholder}."
    summary_template_mixed_disallowed = "Summary for {employee.name}: {performance.rating}, but also {performance.feedback}." # Contains disallowed {performance.feedback}
    summary_template_emp_id = "Summary for {employee.name}: EmpID {employee.emp_id}." # {employee.emp_id} is not in allowed_data_for_template for generate_employee_summary

    test_cases_generate = [
        ("E101", summary_template_valid, "Valid summary"),
        ("E102", summary_template_valid, "Valid summary for another employee"),
        ("E101", summary_template_sensitive_feedback, "Sensitive feedback placeholder"),
        ("E101", summary_template_sensitive_bonus, "Sensitive bonus placeholder"),
        ("E101", summary_template_unknown, "Unknown placeholder"),
        ("E101", summary_template_mixed_disallowed, "Mixed disallowed (feedback)"),
        ("E101", summary_template_emp_id, "Disallowed placeholder (emp_id) for this summary type"),
        ("E999", summary_template_valid, "Non-existent employee ID"),
    ]

    for emp_id, template_str, desc in test_cases_generate:
        print(f"\nAttempting to generate summary for {emp_id} ({desc}): \"{template_str}\"")
        try:
            summary = generate_employee_summary(emp_id, template_str)
            print(f"Generated Summary: {summary}")
        except ValueError as e:
            print(f"Error: {e}")
