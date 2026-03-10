import string

class Employee:
    """Represents an employee with their professional details."""
    def __init__(self, name: str, position: str, department: str, salary: float, personal_email: str):
        """
        Initialize an Employee object.

        Args:
            name: The employee's full name.
            position: The employee's job position.
            department: The department the employee belongs to.
            salary: The employee's salary (considered sensitive).
            personal_email: The employee's personal email (considered sensitive).
        """
        self.name = name
        self.position = position
        self.department = department
        self.salary = salary # Sensitive data
        self.personal_email = personal_email # Sensitive data

# Global list of employee objects.
employees = [
    Employee("Alice Smith", "Software Engineer", "IT", 85000, "alice@personal.com"),
    Employee("Bob Johnson", "Project Manager", "IT", 95000, "bob@personal.com"),
    Employee("Carol Williams", "Data Analyst", "IT", 75000, "carol@personal.com")
]

# Set of attribute names considered sensitive and not allowed in templates.
SENSITIVE_ATTRIBUTES = {'salary', 'personal_email'}

def _validate_template_syntax_and_sensitive_data(format_template: str):
    """
    Validate the format template.

    Checks for:
    1.  Correct Python string format syntax.
    2.  Unauthorized attempts to access sensitive employee attributes
        (e.g., '{employee.salary}').
    3.  Malformed employee attribute access (e.g., '{employee.attr.subattr}').

    Args:
        format_template: The string template to validate.

    Raises:
        ValueError: If any validation fails, detailing the specific error.
    """
    fmt = string.Formatter()
    parsed_template_field_names = []

    # Attempt to parse the template string to identify all format fields.
    # string.Formatter().parse() iterates over the template and returns
    # tuples of (literal_text, field_name, format_spec, conversion).
    try:
        for _, field_name, _, _ in fmt.parse(format_template):
            if field_name: # Only store actual field names, not literal text parts.
                parsed_template_field_names.append(field_name)
    except ValueError as e:
        # Guard clause: Handles malformed format strings (e.g., unclosed braces).
        # This is a primary check for template structural integrity.
        raise ValueError(f"Invalid format template: Malformed syntax - {e}") from e

    # Iterate through all identified field names from the template.
    for field_name in parsed_template_field_names:
        parts = field_name.split('.')

        # Check for placeholders specifically targeting employee attributes (e.g., '{employee.name}').
        if len(parts) == 2 and parts[0] == 'employee':
            attribute_name = parts[1]
            # Guard clause: Security check for sensitive data.
            # If the attribute is in SENSITIVE_ATTRIBUTES, raise an error.
            if attribute_name in SENSITIVE_ATTRIBUTES:
                raise ValueError(
                    f"Attempt to access sensitive attribute '{attribute_name}' in format template."
                )
        # Check for invalid 'employee' attribute access patterns.
        # Examples: '{employee.attr.subattr}' (too many parts) or '{employee.}' (empty attribute).
        elif field_name.startswith("employee.") and (len(parts) != 2 or not parts[1]):
            # Guard clause: Ensures that if 'employee.' is used, it's followed by a valid attribute name.
            raise ValueError(
                f"Invalid employee attribute access: '{field_name}'. Expected format 'employee.attribute'."
            )
        # Other placeholders (e.g., '{foo}', or even '{employee}' for str(employee)) are not
        # explicitly checked here for sensitivity, as they don't fit the 'employee.attribute'
        # pattern for specific data field access. '{employee}' would attempt to stringify the
        # employee object. Other arbitrary placeholders like '{foo}' would lead to a
        # KeyError during the .format() call if 'foo' is not provided; this is handled
        # in _format_employee_entry.

def _filter_employees_by_department(all_employees: list[Employee], dept_name: str) -> list[Employee]:
    """
    Filter a list of employees by their department name.

    Args:
        all_employees: A list of Employee objects.
        dept_name: The name of the department to filter by.

    Returns:
        A new list containing only employees from the specified department.
    """
    # Uses a list comprehension for concise filtering.
    return [emp for emp in all_employees if emp.department == dept_name]

def _format_employee_entry(employee: Employee, format_template: str) -> str:
    """
    Format a single employee's details using the provided template.

    Args:
        employee: The Employee object whose details are to be formatted.
        format_template: The string template to use for formatting.

    Returns:
        A string with the employee's details formatted according to the template.

    Raises:
        ValueError: If the template references a non-existent attribute,
                    uses an invalid placeholder key, or if any other
                    formatting error occurs.
    """
    try:
        # Attempt to format the string with the provided employee object.
        # The 'employee' keyword makes the employee object available in the template,
        # e.g., as {employee.name}.
        return format_template.format(employee=employee)
    except AttributeError as e:
        # Guard clause: Handles templates referencing non-existent attributes on the Employee object.
        # This occurs if format_template is, e.g., "{employee.non_existent_attribute}".
        # This is a key error management point for template content.
        raise ValueError(f"Error formatting directory entry: Invalid attribute in template - {e}") from e
    except KeyError as e:
        # Guard clause: Handles templates using placeholder keys not provided in .format().
        # This occurs if format_template is, e.g., "{foo}" but only "employee" is passed to .format(),
        # or if positional placeholders like "{0.name}" are used with keyword arguments.
        # This is another key error management point for template content.
        raise ValueError(f"Error formatting directory entry: Invalid placeholder key in template - {e}") from e
    except Exception as e:
        # Catch-all: For any other unexpected errors during the .format() call.
        # This ensures robustness against unforeseen issues during string formatting.
        raise ValueError(f"An unexpected error occurred during formatting: {e}") from e

def build_team_directory(dept_name: str, format_template: str) -> str:
    """
    Generate an employee directory listing for a specified department.

    The directory is formatted based on a given template string.
    Placeholders like '{employee.name}' or '{employee.position}' in the
    template are replaced with actual employee data.

    Args:
        dept_name: The department whose employees' information is to be included.
        format_template: The template for formatting employee details.

    Returns:
        A string containing the formatted directory. Each employee's entry
        is on a new line.

    Raises:
        ValueError: If the template attempts to access sensitive data (e.g., salary),
                    is syntactically malformed, references non-existent attributes,
                    or if any other error arises during formatting.
    """
    # Step 1: Validate the template for syntax issues and sensitive data access.
    # This is a crucial pre-processing step to ensure template integrity and security.
    # Raises ValueError if any issues are found.
    _validate_template_syntax_and_sensitive_data(format_template)

    # Step 2: Filter the global list of employees to get only those in the specified department.
    team_employees = _filter_employees_by_department(employees, dept_name)

    # Step 3: Format each employee's entry using the validated template.
    directory_entries = []
    for emp in team_employees:
        # _format_employee_entry handles potential errors during individual formatting,
        # such as AttributeError for invalid fields in the template.
        entry = _format_employee_entry(emp, format_template)
        directory_entries.append(entry)
            
    # Step 4: Join all formatted entries into a single string, separated by newlines.
    return "\n".join(directory_entries)
