from typing import Dict
import re # Moved import re higher to be grouped with other imports

class Employee:
    def __init__(self, emp_id: str, name: str, position: str, department: str):
        self.emp_id = emp_id
        self.name = name
        self.position = position
        self.department = department

    def __repr__(self) -> str:
        return f"Employee(emp_id='{self.emp_id}', name='{self.name}', position='{self.position}', department='{self.department}')"

class Performance:
    def __init__(self, employee: Employee, rating: float, feedback: str, bonus: float):
        self.employee = employee
        self.rating = rating
        self.feedback = feedback # This attribute will be used for the {performance.review} placeholder
        self.bonus = bonus

    def __repr__(self) -> str:
        return f"Performance(employee='{self.employee.name}', rating={self.rating}, feedback='{self.feedback[:30]}...', bonus={self.bonus})"

# Predefined dictionaries of employees and their performances
employees: Dict[str, Employee] = {
    "E101": Employee("E101", "John Doe", "Senior Software Engineer", "R&D"),
    "E102": Employee("E102", "Jane Smith", "Business Analyst", "Operations"),
}

performances: Dict[str, Performance] = {
    "E101": Performance(employees["E101"], 4.3, "Exceeded expectations in multiple projects. Strong technical skills and leadership.", 2000.0),
    "E102": Performance(employees["E102"], 3.8, "Consistently met expectations. Good analytical skills.", 1500.0),
}
# The old get_performance_data function and _employee_performances dictionary are removed by this replacement.

# Define allowed and restricted placeholders
ALLOWED_PLACEHOLDERS = {
    "employee.name",
    "employee.id",
    "employee.department",
    "performance.rating",
    "performance.review"
}

RESTRICTED_PLACEHOLDERS = {
    "employee.salary",  # Example of a restricted field
    "employee.feedback", # As per user request
    "performance.bonus", # As per user request
    "employee.ssn"
}

def validate_summary_template(template_string: str) -> bool:
    """
    Validates a summary template string for allowed placeholders.

    Args:
        template_string: The template string to validate.

    Returns:
        True if all placeholders are valid.

    Raises:
        ValueError: If any placeholder is invalid or restricted.
    """
    placeholders = re.findall(r"\{(.+?)\}", template_string)

    for placeholder in placeholders:
        if placeholder in RESTRICTED_PLACEHOLDERS:
            raise ValueError(f"Placeholder '{{{placeholder}}}' is restricted.")
        if placeholder not in ALLOWED_PLACEHOLDERS:
            raise ValueError(f"Placeholder '{{{placeholder}}}' is not a valid placeholder.")
            
    return True

def build_employee_summary(emp_key: str, template_str: str) -> str:
    """
    Generates a formatted performance summary for an employee using a template string.

    Args:
        emp_key: The string key of the employee (e.g., "E101").
        template_str: The template string with placeholders like {employee.name}
                      or {performance.rating}.

    Returns:
        A formatted summary string with placeholders replaced by actual data.

    Raises:
        ValueError: If the employee ID does not exist in employee records,
                    or if the template string contains invalid or restricted placeholders.
    """
    # First, validate the template string for allowed/restricted placeholders.
    # This will raise a ValueError if the template is invalid.
    validate_summary_template(template_str)

    # Check if the employee exists
    if emp_key not in employees:
        raise ValueError(f"Employee with ID '{emp_key}' not found.")
    
    # Check if performance data exists for the employee
    if emp_key not in performances:
        # This situation implies inconsistent data, but good to check.
        raise ValueError(f"Performance data for employee ID '{emp_key}' not found.")

    employee = employees[emp_key]
    performance = performances[emp_key]

    # Prepare a dictionary with data that can be used for formatting.
    # This maps the allowed placeholder keys to their corresponding data.
    data_for_formatting = {
        "employee.name": employee.name,
        "employee.id": employee.emp_id,
        "employee.department": employee.department,
        "performance.rating": performance.rating,
        "performance.review": performance.feedback,  # Map {performance.review} to the feedback attribute
    }

    summary = template_str
    
    # Find all placeholders in the template string
    placeholders_in_template = re.findall(r"\{(.+?)\}", summary)
    
    for placeholder_key in placeholders_in_template:
        # validate_summary_template has already ensured that placeholder_key is in ALLOWED_PLACEHOLDERS
        # and not in RESTRICTED_PLACEHOLDERS.
        # We now replace it with the actual data.
        if placeholder_key in data_for_formatting:
            summary = summary.replace(f"{{{placeholder_key}}}", str(data_for_formatting[placeholder_key]))
        # No 'else' is strictly needed here because validate_summary_template should have caught
        # any placeholder_key that isn't in ALLOWED_PLACEHOLDERS.
        # However, if data_for_formatting was somehow incomplete for an allowed placeholder,
        # this check `if placeholder_key in data_for_formatting:` adds robustness.

    return summary


if __name__ == '__main__':
    # The old examples for get_performance_data are removed as the function itself is removed.
    
    # Template Validation Examples (still relevant and test validate_summary_template)
    print("--- Template Validation Examples ---")
    valid_template_1 = "Employee: {employee.name}, Rating: {performance.rating}"
    valid_template_2 = "ID: {employee.id} - Review: {performance.review}"
    invalid_template_1 = "Employee: {employee.name}, Salary: {employee.salary}" # Restricted
    invalid_template_2 = "Employee: {employee.name}, Feedback: {employee.feedback}" # Restricted
    invalid_template_3 = "Performance Bonus: {performance.bonus}" # Restricted
    invalid_template_4 = "Employee: {employee.name}, Unknown: {employee.unknown_field}" # Not allowed
    invalid_template_5 = "Details: {performance.summary}" # Not allowed

    templates_to_test = {
        "Valid Template 1": valid_template_1,
        "Valid Template 2": valid_template_2,
        "Invalid Template 1 (salary)": invalid_template_1,
        "Invalid Template 2 (feedback)": invalid_template_2,
        "Invalid Template 3 (bonus)": invalid_template_3,
        "Invalid Template 4 (unknown_field)": invalid_template_4,
        "Invalid Template 5 (summary)": invalid_template_5,
    }

    for name, template in templates_to_test.items():
        try:
            validate_summary_template(template)
            print(f"'{name}': '{template}' - PASSED validation.")
        except ValueError as e:
            print(f"'{name}': '{template}' - FAILED validation: {e}")

    print("\n--- Employee Summary Generation Examples ---")
    
    # Test cases for build_employee_summary
    summary_template_valid = "Summary for {employee.name} (ID: {employee.id}, Dept: {employee.department}): Rating {performance.rating}. Review: {performance.review}"
    summary_template_minimal = "Employee: {employee.name}."
    summary_template_restricted = "Employee: {employee.name}, Bonus: {performance.bonus}" # Uses a restricted placeholder
    summary_template_invalid_ph = "Employee: {employee.name}, Status: {employee.status}" # Uses an invalid (not allowed) placeholder
    summary_template_non_string_replacement = "Rating for {employee.name}: {performance.rating}"


    test_build_cases = [
        ("E101", summary_template_valid, "Valid full summary for E101"),
        ("E102", summary_template_valid, "Valid full summary for E102"),
        ("E101", summary_template_minimal, "Valid minimal summary for E101"),
        ("E102", summary_template_non_string_replacement, "Valid summary with float replacement for E102"),
        ("E103", summary_template_valid, "Attempt summary for non-existent employee E103"),
        ("E101", summary_template_restricted, "Attempt summary with restricted placeholder (bonus) for E101"),
        ("E101", summary_template_invalid_ph, "Attempt summary with invalid placeholder (status) for E101"),
    ]

    for emp_key, template, desc in test_build_cases:
        print(f"\nTest Case: '{desc}'")
        print(f"  Employee Key: '{emp_key}', Template: '{template}'")
        try:
            summary = build_employee_summary(emp_key, template)
            print(f"  SUCCESS: Generated Summary: '{summary}'")
        except ValueError as e:
            print(f"  FAILED: Error: {e}")

    # Example of a successful direct call
    print("\n--- Direct Call Example (Successful) ---")
    try:
        emp_id_to_test = "E101"
        template_to_test = "Name: {employee.name}, Rating: {performance.rating}."
        print(f"Generating summary for '{emp_id_to_test}' with template: '{template_to_test}'")
        generated_summary = build_employee_summary(emp_id_to_test, template_to_test)
        print("Generated Summary:")
        print(generated_summary)
    except ValueError as e:
        print(f"Error generating summary: {e}")
