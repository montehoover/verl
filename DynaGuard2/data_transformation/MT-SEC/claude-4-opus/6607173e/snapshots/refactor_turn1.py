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

def create_employee_directory(department: str, template: str) -> str:
    try:
        # Filter employees by department
        dept_employees = [emp for emp in employees if emp.department == department]
        
        if not dept_employees:
            return ""
        
        # Check for sensitive information in template
        sensitive_attrs = ['salary', 'personal_email']
        for attr in sensitive_attrs:
            if f'{{employee.{attr}}}' in template:
                raise ValueError(f"Access to sensitive information '{attr}' is not allowed")
        
        # Build directory string
        directory_lines = []
        for emp in dept_employees:
            line = template
            
            # Replace placeholders with actual values
            if '{employee.name}' in line:
                line = line.replace('{employee.name}', emp.name)
            if '{employee.position}' in line:
                line = line.replace('{employee.position}', emp.position)
            if '{employee.department}' in line:
                line = line.replace('{employee.department}', emp.department)
            
            # Check for any remaining placeholders that might be sensitive
            if '{employee.' in line:
                raise ValueError("Invalid or unauthorized employee attribute access")
            
            directory_lines.append(line)
        
        return '\n'.join(directory_lines)
    
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        else:
            raise ValueError(f"An error occurred while creating the directory: {str(e)}")
