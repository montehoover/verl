from typing import Dict

class Performance:
    """Represents an employee's performance data."""
    def __init__(self, rating: float, review: str):
        self.rating = rating
        self.review = review

    def __repr__(self) -> str:
        return f"Performance(rating={self.rating}, review='{self.review}')"

# Predefined dictionary of employee performances
# In a real application, this data might come from a database or an external service.
_employee_performances: Dict[int, Performance] = {
    101: Performance(rating=4.5, review="Exceeds expectations in all areas."),
    102: Performance(rating=3.0, review="Meets expectations, room for growth in project management."),
    103: Performance(rating=4.0, review="Strong performer, consistently delivers high-quality work."),
    104: Performance(rating=2.5, review="Needs improvement in communication skills."),
}

def get_performance_data(employee_id: int) -> Performance:
    """
    Fetches the performance data for a given employee ID.

    Args:
        employee_id: The ID of the employee.

    Returns:
        The Performance object for the employee.

    Raises:
        ValueError: If the employee ID does not exist in the performance records.
    """
    if employee_id in _employee_performances:
        return _employee_performances[employee_id]
    else:
        raise ValueError(f"No performance data found for employee ID: {employee_id}")

import re

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


if __name__ == '__main__':
    # Example usage:
    try:
        perf_101 = get_performance_data(101)
        print(f"Employee 101 Performance: {perf_101}")

        perf_102 = get_performance_data(102)
        print(f"Employee 102 Performance: {perf_102}")

        # Example of an employee ID that does not exist
        perf_999 = get_performance_data(999)
        print(f"Employee 999 Performance: {perf_999}")
    except ValueError as e:
        print(f"Error: {e}")

    try:
        # Another non-existent ID
        perf_105 = get_performance_data(105)
        print(f"Employee 105 Performance: {perf_105}")
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Template Validation Examples ---")
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
