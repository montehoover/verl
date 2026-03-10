import re

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
    # Check for sensitive data access in the template
    sensitive_fields = ['salary', 'personal_email']
    for field in sensitive_fields:
        if field in format_template:
            raise ValueError(f"Access to sensitive data '{field}' is not allowed")
    
    # Filter employees by department
    dept_employees = [emp for emp in employees if emp.department == dept_name]
    
    # If no employees found in the department, return empty string
    if not dept_employees:
        return ""
    
    # Format each employee's information
    formatted_entries = []
    for employee in dept_employees:
        try:
            # Replace placeholders in the template
            formatted_entry = format_template
            
            # Find all placeholders like {employee.field}
            placeholders = re.findall(r'\{employee\.(\w+)\}', format_template)
            
            for placeholder in placeholders:
                # Check again for sensitive fields
                if placeholder in sensitive_fields:
                    raise ValueError(f"Access to sensitive data '{placeholder}' is not allowed")
                
                # Get the attribute value
                if hasattr(employee, placeholder):
                    value = getattr(employee, placeholder)
                    formatted_entry = formatted_entry.replace(f'{{employee.{placeholder}}}', str(value))
                else:
                    raise ValueError(f"Employee has no attribute '{placeholder}'")
            
            formatted_entries.append(formatted_entry)
            
        except Exception as e:
            raise ValueError(f"Error formatting employee data: {str(e)}")
    
    # Join all entries with newlines
    return '\n'.join(formatted_entries)
