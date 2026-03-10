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

SENSITIVE_ATTRIBUTES = {'salary', 'personal_email'}

def _validate_template_syntax_and_sensitive_data(format_template: str):
    """
    Validates the format template for syntax errors and unauthorized access
    to sensitive employee attributes.
    Raises ValueError if validation fails.
    """
    fmt = string.Formatter()
    parsed_template_field_names = []
    try:
        for _, field_name, _, _ in fmt.parse(format_template):
            if field_name:
                parsed_template_field_names.append(field_name)
    except ValueError as e: # Handles malformed format strings like unclosed braces
        raise ValueError(f"Invalid format template: Malformed syntax - {e}") from e

    for field_name in parsed_template_field_names:
        parts = field_name.split('.')
        if len(parts) == 2 and parts[0] == 'employee':
            attribute_name = parts[1]
            if attribute_name in SENSITIVE_ATTRIBUTES:
                raise ValueError(
                    f"Attempt to access sensitive attribute '{attribute_name}' in format template."
                )
        elif field_name.startswith("employee.") and (len(parts) != 2 or not parts[1]):
            raise ValueError(
                f"Invalid employee attribute access: '{field_name}'. Expected format 'employee.attribute'."
            )

def _filter_employees_by_department(all_employees: list[Employee], dept_name: str) -> list[Employee]:
    """Filters a list of employees by department name."""
    return [emp for emp in all_employees if emp.department == dept_name]

def _format_employee_entry(employee: Employee, format_template: str) -> str:
    """Formats a single employee's details using the template."""
    try:
        return format_template.format(employee=employee)
    except AttributeError as e:
        raise ValueError(f"Error formatting directory entry: Invalid attribute in template - {e}") from e
    except KeyError as e:
        raise ValueError(f"Error formatting directory entry: Invalid placeholder key in template - {e}") from e
    except Exception as e:
        raise ValueError(f"An unexpected error occurred during formatting: {e}") from e

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
    _validate_template_syntax_and_sensitive_data(format_template)

    team_employees = _filter_employees_by_department(employees, dept_name)

    directory_entries = []
    for emp in team_employees:
        entry = _format_employee_entry(emp, format_template)
        directory_entries.append(entry)
            
    return "\n".join(directory_entries)
