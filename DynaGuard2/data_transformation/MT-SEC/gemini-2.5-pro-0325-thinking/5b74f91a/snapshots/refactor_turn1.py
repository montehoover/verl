import string

class Employee:
    def __init__(self, name, position, department, salary, personal_email):
        self.name = name
        self.position = position
        self.department = department
        self.salary = salary
        self.personal_email = personal_email

employees = [
    Employee("Alice Smith", "Software Engineer", "IT", 85000, "alice@personal.com"),
    Employee("Bob Johnson", "Project Manager", "IT", 95000, "bob@personal.com"),
    Employee("Carol Williams", "Data Analyst", "IT", 75000, "carol@personal.com")
]

def build_team_directory(dept_name: str, format_template: str) -> str:
    """
    Generates an employee directory listing for a specified department,
    formatted based on a given template string.

    Args:
        dept_name: The department whose employees' information needs to be included.
        format_template: The template for formatting the employee details.
                         Placeholders like '{employee.name}' or '{employee.position}'
                         will be replaced with actual employee data.

    Returns:
        A formatted string directory based on the employees belonging to
        the specified department. Each employee's entry is on a new line.

    Raises:
        ValueError: If an attempt to access sensitive data (salary, personal_email)
                    is made in the template, if the template is malformed,
                    or if any other error arises during formatting.
    """
    SENSITIVE_ATTRIBUTES = {'salary', 'personal_email'}
    
    fmt = string.Formatter()
    parsed_template_field_names = []
    try:
        # Iterate through parsed fields to check for errors or extract field names
        for _, field_name, _, _ in fmt.parse(format_template):
            if field_name:
                parsed_template_field_names.append(field_name)
    except ValueError as e: # Handles malformed format strings like unclosed braces
        raise ValueError(f"Invalid format template: Malformed syntax - {e}") from e

    for field_name in parsed_template_field_names:
        parts = field_name.split('.')
        # Check for placeholders like '{employee.attribute}'
        if len(parts) == 2 and parts[0] == 'employee':
            attribute_name = parts[1]
            if attribute_name in SENSITIVE_ATTRIBUTES:
                raise ValueError(
                    f"Attempt to access sensitive attribute '{attribute_name}' in format template."
                )
        # Check for invalid 'employee' attribute access like '{employee.attr.subattr}' or '{employee.}'
        elif field_name.startswith("employee.") and (len(parts) != 2 or not parts[1]):
            raise ValueError(
                f"Invalid employee attribute access: '{field_name}'. Expected format 'employee.attribute'."
            )
        # Other placeholders (e.g., '{foo}', or even '{employee}' for str(employee)) are not checked here for sensitivity,
        # as they don't fit the 'employee.attribute' pattern for specific data field access.
        # '{employee}' will attempt to stringify the employee object.
        # Other arbitrary placeholders like '{foo}' will lead to KeyError during formatting if not 'employee'.

    directory_entries = []
    team_employees = [emp for emp in employees if emp.department == dept_name]

    for emp in team_employees:
        try:
            formatted_entry = format_template.format(employee=emp)
            directory_entries.append(formatted_entry)
        except AttributeError as e:
            # Handles {employee.non_existent_attribute}
            raise ValueError(f"Error formatting directory entry: Invalid attribute in template - {e}") from e
        except KeyError as e:
            # Handles placeholders like "{foo}" when only "employee" is provided as a keyword argument to format.
            # Or if format_template was "{0.name}" but .format(employee=emp) was called.
            raise ValueError(f"Error formatting directory entry: Invalid placeholder key in template - {e}") from e
        except Exception as e: # Catch other potential errors during .format()
            raise ValueError(f"An unexpected error occurred during formatting: {e}") from e
            
    return "\n".join(directory_entries)
