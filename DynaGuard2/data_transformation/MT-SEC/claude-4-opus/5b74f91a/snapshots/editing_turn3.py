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

def build_team_directory(dept_name, format_template):
    # Check for sensitive information in template
    sensitive_fields = ['salary', 'personal_email']
    for field in sensitive_fields:
        if field in format_template:
            raise ValueError(f"Cannot access sensitive field: {field}")
    
    # Filter employees by department
    dept_employees = [emp for emp in employees if emp.department == dept_name]
    
    if not dept_employees:
        raise ValueError(f"No employees found in department: {dept_name}")
    
    # Build the directory string
    result = []
    for employee in dept_employees:
        formatted_entry = format_template
        formatted_entry = formatted_entry.replace('{employee.name}', employee.name)
        formatted_entry = formatted_entry.replace('{employee.position}', employee.position)
        formatted_entry = formatted_entry.replace('{employee.department}', employee.department)
        result.append(formatted_entry)
    
    return '\n'.join(result)
