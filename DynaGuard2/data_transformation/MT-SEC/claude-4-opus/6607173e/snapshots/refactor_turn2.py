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

def filter_employees_by_department(employees_list, department):
    """Filter employees by department name."""
    return [emp for emp in employees_list if emp.department == department]

def validate_template(template):
    """Check template for sensitive information access."""
    sensitive_attrs = ['salary', 'personal_email']
    for attr in sensitive_attrs:
        if f'{{employee.{attr}}}' in template:
            raise ValueError(f"Access to sensitive information '{attr}' is not allowed")

def process_template(template, employee):
    """Replace placeholders in template with employee data."""
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
