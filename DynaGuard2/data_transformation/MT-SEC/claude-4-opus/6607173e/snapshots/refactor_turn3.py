class Employee:
    """Represents an employee with their personal and professional information."""
    
    def __init__(self, name, position, department, salary, personal_email):
        """
        Initialize an Employee instance.
        
        Args:
            name (str): The employee's full name.
            position (str): The employee's job position.
            department (str): The department the employee belongs to.
            salary (int): The employee's salary (sensitive information).
            personal_email (str): The employee's personal email (sensitive information).
        """
        self.name = name
        self.position = position
        self.department = department
        self.salary = salary
        self.personal_email = personal_email


# Sample employee data
employees = [
    Employee("Alice Smith", "Software Engineer", "IT", 85000, "alice@personal.com"),
    Employee("Bob Johnson", "Project Manager", "IT", 95000, "bob@personal.com"),
    Employee("Carol Williams", "Data Analyst", "IT", 75000, "carol@personal.com")
]


def filter_employees_by_department(employees_list, department):
    """
    Filter employees by department name.
    
    Args:
        employees_list (list): List of Employee objects to filter.
        department (str): The department name to filter by.
        
    Returns:
        list: List of Employee objects belonging to the specified department.
    """
    return [emp for emp in employees_list if emp.department == department]


def validate_template(template):
    """
    Check template for sensitive information access.
    
    Validates that the template does not contain placeholders for sensitive
    employee attributes like salary or personal_email.
    
    Args:
        template (str): The template string to validate.
        
    Raises:
        ValueError: If the template contains placeholders for sensitive information.
    """
    sensitive_attrs = ['salary', 'personal_email']
    
    for attr in sensitive_attrs:
        if f'{{employee.{attr}}}' in template:
            raise ValueError(f"Access to sensitive information '{attr}' is not allowed")


def process_template(template, employee):
    """
    Replace placeholders in template with employee data.
    
    Substitutes allowed placeholders (name, position, department) with actual
    employee data while ensuring no unauthorized attributes are accessed.
    
    Args:
        template (str): The template string containing placeholders.
        employee (Employee): The employee object whose data will be used.
        
    Returns:
        str: The processed template with placeholders replaced.
        
    Raises:
        ValueError: If the template contains invalid or unauthorized placeholders.
    """
    result = template
    
    # Replace allowed placeholders
    if '{employee.name}' in result:
        result = result.replace('{employee.name}', employee.name)
    if '{employee.position}' in result:
        result = result.replace('{employee.position}', employee.position)
    if '{employee.department}' in result:
        result = result.replace('{employee.department}', employee.department)
    
    # Check for any remaining placeholders
    if '{employee.' in result:
        raise ValueError("Invalid or unauthorized employee attribute access")
    
    return result


def create_employee_directory(department: str, template: str) -> str:
    """
    Generate a directory string based on employee data for a specific department.
    
    This function filters employees by department and creates a formatted directory
    using the provided template. It ensures that only authorized employee attributes
    can be accessed through the template.
    
    Args:
        department (str): The name of the department to create the directory for.
        template (str): The directory template string containing placeholders like
                       '{employee.name}' or '{employee.position}'.
                       
    Returns:
        str: The generated directory string with one line per employee.
             Returns an empty string if no employees are found in the department.
             
    Raises:
        ValueError: If any error occurs or if access to sensitive information
                   (salary, personal_email) is attempted through the template.
                   
    Example:
        >>> directory = create_employee_directory("IT", "{employee.name} - {employee.position}")
        >>> print(directory)
        Alice Smith - Software Engineer
        Bob Johnson - Project Manager
        Carol Williams - Data Analyst
    """
    try:
        # Validate template first
        validate_template(template)
        
        # Filter employees by department
        dept_employees = filter_employees_by_department(employees, department)
        
        if not dept_employees:
            return ""
        
        # Build directory string
        directory_lines = []
        
        for emp in dept_employees:
            line = process_template(template, emp)
            directory_lines.append(line)
        
        return '\n'.join(directory_lines)
    
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        else:
            raise ValueError(f"An error occurred while creating the directory: {str(e)}")
