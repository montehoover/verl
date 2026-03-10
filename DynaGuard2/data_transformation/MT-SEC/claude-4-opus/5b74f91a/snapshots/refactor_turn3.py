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


def validate_template(format_template: str) -> None:
    """Check if the template contains sensitive fields.
    
    This function validates that the template does not attempt to access
    sensitive employee data such as salary or personal email addresses.
    
    Args:
        format_template: The template string to validate.
        
    Raises:
        ValueError: If the template contains references to sensitive fields.
    """
    # Define fields that should not be accessible through templates
    sensitive_fields = ['salary', 'personal_email']
    
    # Check each sensitive field for various possible template formats
    for field in sensitive_fields:
        # Check for both direct references and bracketed template syntax
        if f'employee.{field}' in format_template or f'{{employee.{field}}}' in format_template:
            raise ValueError(f"Access to sensitive data '{field}' is not allowed")


def filter_employees_by_department(employees_list: list, dept_name: str) -> list:
    """Filter employees by department name.
    
    Returns a list of employees who belong to the specified department.
    
    Args:
        employees_list: List of Employee objects to filter.
        dept_name: Name of the department to filter by.
        
    Returns:
        List of Employee objects belonging to the specified department.
    """
    return [emp for emp in employees_list if emp.department == dept_name]


def format_employee_entry(employee: Employee, format_template: str) -> str:
    """Format a single employee entry based on the template.
    
    Replaces template placeholders with actual employee data. Only allows
    access to non-sensitive fields: name, position, and department.
    
    Args:
        employee: Employee object containing the data to format.
        format_template: Template string with placeholders.
        
    Returns:
        Formatted string with placeholders replaced by employee data.
        
    Raises:
        ValueError: If invalid template placeholders are found.
    """
    # Start with a copy of the template
    entry = format_template
    
    # Replace allowed placeholders with actual employee data
    entry = entry.replace('{employee.name}', employee.name)
    entry = entry.replace('{employee.position}', employee.position)
    entry = entry.replace('{employee.department}', employee.department)
    
    # Guard clause: Check for any remaining employee placeholders
    # This catches attempts to access fields we haven't explicitly allowed
    if '{employee.' in entry:
        raise ValueError("Invalid template placeholder")
    
    return entry


def build_team_directory(dept_name: str, format_template: str) -> str:
    """Generate an employee directory listing for a specified department.
    
    This function creates a formatted directory of employees belonging to
    a specific department using a provided template. It prevents access
    to sensitive information like salary and personal email.
    
    Args:
        dept_name: The department whose employees' information needs to be included.
        format_template: The template for formatting the employee details.
        
    Returns:
        A formatted string directory based on the employees belonging to the
        specified department, with each employee on a new line.
        
    Raises:
        ValueError: If an attempt to access sensitive data is made or if
                   an error arises during template processing.
    """
    # First, validate that the template doesn't contain sensitive field references
    validate_template(format_template)
    
    # Filter to get only employees from the specified department
    dept_employees = filter_employees_by_department(employees, dept_name)
    
    # Guard clause: If no employees found, return empty string
    if not dept_employees:
        return ""
    
    # Process each employee and build the directory entries
    directory_entries = []
    
    for employee in dept_employees:
        try:
            # Format the current employee's entry using the template
            entry = format_employee_entry(employee, format_template)
            directory_entries.append(entry)
            
        except Exception as e:
            # Re-raise any errors with additional context
            raise ValueError(f"Error processing template: {str(e)}")
    
    # Join all entries with newlines to create the final directory
    return '\n'.join(directory_entries)
