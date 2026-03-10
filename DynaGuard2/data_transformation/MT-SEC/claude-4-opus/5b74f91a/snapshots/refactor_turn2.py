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
    """Check if the template contains sensitive fields."""
    sensitive_fields = ['salary', 'personal_email']
    for field in sensitive_fields:
        if f'employee.{field}' in format_template or f'{{employee.{field}}}' in format_template:
            raise ValueError(f"Access to sensitive data '{field}' is not allowed")

def filter_employees_by_department(employees_list: list, dept_name: str) -> list:
    """Filter employees by department name."""
    return [emp for emp in employees_list if emp.department == dept_name]

def format_employee_entry(employee: Employee, format_template: str) -> str:
    """Format a single employee entry based on the template."""
    entry = format_template
    entry = entry.replace('{employee.name}', employee.name)
    entry = entry.replace('{employee.position}', employee.position)
    entry = entry.replace('{employee.department}', employee.department)
    
    # Check if there are any remaining placeholders that might be trying to access other attributes
    if '{employee.' in entry:
        raise ValueError("Invalid template placeholder")
    
    return entry

def build_team_directory(dept_name: str, format_template: str) -> str:
    # Validate template for sensitive data
    validate_template(format_template)
    
    # Filter employees by department
    dept_employees = filter_employees_by_department(employees, dept_name)
    
    # Build the directory
    directory_entries = []
    for employee in dept_employees:
        try:
            entry = format_employee_entry(employee, format_template)
            directory_entries.append(entry)
        except Exception as e:
            raise ValueError(f"Error processing template: {str(e)}")
    
    return '\n'.join(directory_entries)
