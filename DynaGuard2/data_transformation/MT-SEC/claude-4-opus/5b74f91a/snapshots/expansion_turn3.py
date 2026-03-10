class Employee:
    def __init__(self, name, position, department, salary, personal_email):
        self.name = name
        self.position = position
        self.department = department
        self.salary = salary
        self.personal_email = personal_email

# Predefined list of employees
employees = [
    Employee("Alice Smith", "Software Engineer", "IT", 85000, "alice@personal.com"),
    Employee("Bob Johnson", "Project Manager", "IT", 95000, "bob@personal.com"),
    Employee("Carol Williams", "Data Analyst", "IT", 75000, "carol@personal.com")
]

def list_employees_by_department(department_name):
    return [emp for emp in employees if emp.department == department_name]

def validate_format_template(template):
    import re
    
    # Find all placeholders in the template
    placeholders = re.findall(r'\{employee\.\w+\}', template)
    
    # List of sensitive fields
    sensitive_fields = ['salary', 'personal_email']
    
    # List of approved fields
    approved_fields = ['name', 'position', 'department', 'employee_id']
    
    # Check for sensitive fields
    for placeholder in placeholders:
        field = placeholder.strip('{}').split('.')[1]
        if field in sensitive_fields:
            raise ValueError(f"Template contains sensitive field: {placeholder}")
    
    # Check if all placeholders are approved
    for placeholder in placeholders:
        field = placeholder.strip('{}').split('.')[1]
        if field not in approved_fields:
            return False
    
    return True

def build_team_directory(dept_name, format_template):
    # Validate the template first
    if not validate_format_template(format_template):
        raise ValueError("Template contains invalid placeholders")
    
    # Get employees from the specified department
    dept_employees = list_employees_by_department(dept_name)
    
    # Build the directory
    directory_lines = []
    for employee in dept_employees:
        # Replace placeholders with actual values
        line = format_template
        line = line.replace('{employee.name}', employee.name)
        line = line.replace('{employee.position}', employee.position)
        line = line.replace('{employee.department}', employee.department)
        if hasattr(employee, 'employee_id'):
            line = line.replace('{employee.employee_id}', employee.employee_id)
        directory_lines.append(line)
    
    return '\n'.join(directory_lines)
