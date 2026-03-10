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
    # Check for sensitive data access attempts
    sensitive_fields = ['salary', 'personal_email']
    for field in sensitive_fields:
        if f'employee.{field}' in format_template or f'{{employee.{field}}}' in format_template:
            raise ValueError(f"Access to sensitive data '{field}' is not allowed")
    
    # Filter employees by department
    dept_employees = [emp for emp in employees if emp.department == dept_name]
    
    # Build the directory
    directory_entries = []
    for employee in dept_employees:
        try:
            # Replace placeholders in the template
            entry = format_template
            entry = entry.replace('{employee.name}', employee.name)
            entry = entry.replace('{employee.position}', employee.position)
            entry = entry.replace('{employee.department}', employee.department)
            
            # Check if there are any remaining placeholders that might be trying to access other attributes
            if '{employee.' in entry:
                raise ValueError("Invalid template placeholder")
                
            directory_entries.append(entry)
        except Exception as e:
            raise ValueError(f"Error processing template: {str(e)}")
    
    return '\n'.join(directory_entries)
