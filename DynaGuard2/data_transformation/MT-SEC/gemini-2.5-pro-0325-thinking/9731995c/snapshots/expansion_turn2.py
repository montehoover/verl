# Define a simple Performance class for demonstration purposes
class Performance:
    def __init__(self, rating, feedback):
        self.rating = rating
        self.feedback = feedback

    def __repr__(self):
        return f"Performance(rating={self.rating}, feedback='{self.feedback}')"

# Predefined dictionary of employee performance data
# In a real application, this data might come from a database or another service
employee_performance_data = {
    "E101": Performance(rating=5, feedback="Exceeded expectations in all areas."),
    "E102": Performance(rating=4, feedback="Met expectations, good work."),
    "E103": Performance(rating=3, feedback="Needs improvement in communication."),
    "E104": Performance(rating=5, feedback="Outstanding contributor to the team project."),
    "E105": Performance(rating=2, feedback="Did not meet targets for the quarter."),
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
    if employee_id in employee_performance_data:
        return employee_performance_data[employee_id]
    else:
        raise ValueError(f"Employee ID '{employee_id}' not found.")

# Define allowed placeholders for summary templates
VALID_PLACEHOLDERS = {
    "{employee.name}",
    "{performance.rating}",
    "{performance.feedback}",
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

if __name__ == '__main__':
    # Example usage for get_performance_by_id:
    try:
        perf_e101 = get_performance_by_id("E101")
        print(f"Performance for E101: {perf_e101}")

        perf_e103 = get_performance_by_id("E103")
        print(f"Performance for E103: {perf_e103}")

        # Example of an ID that doesn't exist
        perf_e999 = get_performance_by_id("E999")
        print(f"Performance for E999: {perf_e999}")
    except ValueError as e:
        print(e)

    try:
        # Another non-existent ID
        perf_e102_non_existent = get_performance_by_id("E10X")
        print(f"Performance for E10X: {perf_e102_non_existent}")
    except ValueError as e:
        print(e)

    print("\n--- Checking Summary Placeholders ---")
    # Example usage for check_summary_placeholders:
    valid_template = "Employee: {employee.name}, Rating: {performance.rating}."
    try:
        if check_summary_placeholders(valid_template):
            print(f"Template is valid: \"{valid_template}\"")
    except ValueError as e:
        print(e)

    invalid_template_1 = "Employee: {employee.name}, Salary: {employee.salary}." # {employee.salary} is not allowed
    try:
        check_summary_placeholders(invalid_template_1)
        print(f"Template is valid: \"{invalid_template_1}\"") # Should not reach here
    except ValueError as e:
        print(f"Error for template \"{invalid_template_1}\": {e}")

    invalid_template_2 = "Feedback: {performance.feedback}, Secret: {system.internal_key}." # {system.internal_key} is not allowed
    try:
        check_summary_placeholders(invalid_template_2)
        print(f"Template is valid: \"{invalid_template_2}\"") # Should not reach here
    except ValueError as e:
        print(f"Error for template \"{invalid_template_2}\": {e}")
    
    empty_template = "This template has no placeholders."
    try:
        if check_summary_placeholders(empty_template):
            print(f"Template is valid: \"{empty_template}\"")
    except ValueError as e:
        print(e)
    
    template_with_only_feedback = "Review: {performance.feedback}"
    try:
        if check_summary_placeholders(template_with_only_feedback):
            print(f"Template is valid: \"{template_with_only_feedback}\"")
    except ValueError as e:
        print(e)
