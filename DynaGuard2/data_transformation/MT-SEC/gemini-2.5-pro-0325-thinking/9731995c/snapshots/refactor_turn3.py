import re
import logging

# Set up a logger for this module
logger = logging.getLogger(__name__)

class Employee:
    def __init__(self, emp_id, name, position, department):
        self.emp_id = emp_id
        self.name = name
        self.position = position
        self.department = department

class Performance:
    def __init__(self, employee, rating, feedback, bonus):
        self.employee = employee
        self.rating = rating
        self.feedback = feedback
        self.bonus = bonus

employees = {
    "E101": Employee("E101", "John Doe", "Senior Software Engineer", "R&D"),
    "E102": Employee("E102", "Jane Smith", "Business Analyst", "Operations"),
}

performances = {
    "E101": Performance(employees["E101"], 4.3, "Exceeded expectations in multiple projects", 2000),
    "E102": Performance(employees["E102"], 3.8, "Consistently met expectations.", 1500),
}

def generate_employee_summary(employee_id: str, template: str) -> str:
    """
    Generates a personalized employee summary string by substituting placeholders
    in a template with actual employee and performance data.

    This function retrieves employee and performance information based on the
    provided `employee_id`. It then parses the `template` string, looking for
    placeholders in the format `{object.attribute}` (e.g., `{employee.name}`,
    `{performance.rating}`). These placeholders are replaced with the
    corresponding data.

    Access to sensitive information like performance feedback or bonus details
    via placeholders is strictly prohibited and will result in a ValueError.

    Args:
        employee_id (str): The unique identifier for the employee.
        template (str): A string containing the summary format with
                        placeholders. For example:
                        "Summary for {employee.name} ({employee.position}): Rating - {performance.rating}."

    Returns:
        str: The summary string with all valid placeholders replaced by
             actual data.

    Raises:
        ValueError:
            - If `employee_id` or `template` is not a string.
            - If an employee with the given `employee_id` is not found.
            - If a performance record for the given `employee_id` is not found.
            - If the template contains placeholders for sensitive data
              (e.g., `{performance.feedback}`, `{performance.bonus}`).
            - If the template contains invalid placeholders (e.g., unknown object
              type, unknown attribute, or unsupported placeholders like
              `{performance.employee}`).
            - If a placeholder attempts to access a conventionally private attribute
              (e.g., `{employee._internal_id}`).
    """
    # Initial input validation (Guard Clauses)
    if not isinstance(employee_id, str):
        # This ValueError will be raised directly, not caught by the try-except below,
        # as it's a fundamental input type error.
        raise ValueError("Employee ID must be a string.")
    if not isinstance(template, str):
        # Similarly, this is a fundamental input type error.
        raise ValueError("Template must be a string.")

    try:
        # Retrieve employee and performance records (Guard Clauses for existence)
        if employee_id not in employees:
            raise ValueError(f"Employee with ID '{employee_id}' not found.")
        employee_record = employees[employee_id]

        if employee_id not in performances:
            raise ValueError(f"Performance record for employee ID '{employee_id}' not found.")
        performance_record = performances[employee_id]

        # Define the replacer function for re.sub
        # This function contains further guard clauses for placeholder validity
        def replacer(match_obj: re.Match) -> str:
            placeholder_object_type = match_obj.group(1)
            placeholder_attribute_key = match_obj.group(2)

            if placeholder_object_type == "employee":
                # Policy check: Prevent access to conventionally private attributes
                if placeholder_attribute_key.startswith('_'):
                    raise ValueError(f"Access to conventionally private attribute '{{employee.{placeholder_attribute_key}}}' is not allowed.")
                
                try:
                    value = getattr(employee_record, placeholder_attribute_key)
                    return str(value)
                except AttributeError:
                    # This error indicates an invalid placeholder attribute for employee
                    raise ValueError(f"Employee has no attribute '{placeholder_attribute_key}'. Invalid placeholder: {{employee.{placeholder_attribute_key}}}")
            
            elif placeholder_object_type == "performance":
                # Policy check: Prevent access to sensitive attributes
                if placeholder_attribute_key in ["feedback", "bonus"]:
                    raise ValueError(f"Access to sensitive performance data ('{placeholder_attribute_key}') is not allowed. Invalid placeholder: {{performance.{placeholder_attribute_key}}}")
                
                # Policy check: Disallow direct access to the nested employee object via performance
                if placeholder_attribute_key == "employee":
                    raise ValueError("Placeholder '{performance.employee}' is not supported. Access employee attributes directly via '{employee.attribute_name}'.")
                
                # Policy check: Prevent access to conventionally private attributes
                if placeholder_attribute_key.startswith('_'):
                    raise ValueError(f"Access to conventionally private attribute '{{performance.{placeholder_attribute_key}}}' is not allowed.")

                try:
                    value = getattr(performance_record, placeholder_attribute_key)
                    return str(value)
                except AttributeError:
                    # This error indicates an invalid placeholder attribute for performance
                    raise ValueError(f"Performance record has no attribute '{placeholder_attribute_key}'. Invalid placeholder: {{performance.{placeholder_attribute_key}}}")
            
            else:
                # Handle unknown object types in placeholders
                raise ValueError(f"Unknown object type '{placeholder_object_type}' in placeholder '{{{placeholder_object_type}.{placeholder_attribute_key}}}'")

        # Perform the substitution using the replacer function
        summary = re.sub(r'{(\w+)\.(\w+)}', replacer, template)
        
        return summary

    except ValueError as ve:
        # Log ValueErrors (from lookups, replacer function, etc.) and re-raise
        logger.error(f"ValueError during summary generation for employee '{employee_id}': {ve}")
        raise
    except Exception as e:
        # Log any other unexpected exceptions and wrap them in a ValueError
        logger.error(f"Unexpected error during summary generation for employee '{employee_id}': {e}", exc_info=True)
        raise ValueError(f"An unexpected error occurred while generating summary for employee '{employee_id}'.") from e
