class Employee:
    def __init__(self, id, name, position):
        self.id = id
        self.name = name
        self.position = position

    def __repr__(self):
        return f"Employee(id={self.id}, name='{self.name}', position='{self.position}')"

# Predefined dictionary of employees
# In a real application, this data might come from a database or an external file.
_employees_data = {
    101: Employee(101, "Alice Wonderland", "Software Engineer"),
    102: Employee(102, "Bob The Builder", "Project Manager"),
    103: Employee(103, "Charlie Brown", "Data Scientist"),
    104: Employee(104, "Diana Prince", "UX Designer"),
}

def get_employee_by_id(employee_id: int) -> Employee:
    """
    Retrieves an employee object by their ID.

    Args:
        employee_id: The ID of the employee to retrieve.

    Returns:
        The Employee object corresponding to the given ID.

    Raises:
        ValueError: If an employee with the given ID does not exist.
    """
    employee = _employees_data.get(employee_id)
    if employee is None:
        raise ValueError(f"Employee with ID {employee_id} not found.")
    return employee

import re

# Define allowed placeholders
ALLOWED_PLACEHOLDERS = {
    "personnel.name",
    "review.rating",
    # Add more allowed placeholders here as needed
    # For example:
    # "personnel.id",
    # "personnel.position",
    # "review.date",
    # "review.feedback",
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

if __name__ == '__main__':
    # Example usage:
    try:
        employee = get_employee_by_id(101)
        print(f"Found employee: {employee}")

        employee_non_existent = get_employee_by_id(999)
        print(f"Found employee: {employee_non_existent}")
    except ValueError as e:
        print(f"Error: {e}")

    try:
        employee_bob = get_employee_by_id(102)
        print(f"Found employee: {employee_bob}")
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
