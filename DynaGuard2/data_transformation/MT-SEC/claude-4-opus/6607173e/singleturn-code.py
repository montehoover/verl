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
    """
    Creates an employee directory string for a given department using a template.
    
    Args:
        department: The name of the department
        template: The directory template string with placeholders like {employee.name}
        
    Returns:
        The generated directory string with one line per employee
        
    Raises:
        ValueError: If access to sensitive information is attempted or any error occurs
    """
    # Check for sensitive information access attempts
    sensitive_fields = ['salary', 'personal_email']
    for field in sensitive_fields:
        if f'{{employee.{field}}}' in template:
            raise ValueError(f"Access to sensitive information '{field}' is not allowed")
    
    # Filter employees by department
    department_employees = [emp for emp in employees if emp.department == department]
    
    # Generate directory lines
    directory_lines = []
    for employee in department_employees:
        try:
            # Replace placeholders in template
            line = template
            
            # Find all placeholders in the format {employee.attribute}
            import re
            placeholders = re.findall(r'\{employee\.(\w+)\}', template)
            
            for placeholder in placeholders:
                # Check if attribute exists and is not sensitive
                if placeholder in sensitive_fields:
                    raise ValueError(f"Access to sensitive information '{placeholder}' is not allowed")
                
                if hasattr(employee, placeholder):
                    value = getattr(employee, placeholder)
                    line = line.replace(f'{{employee.{placeholder}}}', str(value))
                else:
                    raise ValueError(f"Invalid employee attribute: {placeholder}")
            
            directory_lines.append(line)
            
        except Exception as e:
            raise ValueError(f"Error processing template: {str(e)}")
    
    return '\n'.join(directory_lines)
