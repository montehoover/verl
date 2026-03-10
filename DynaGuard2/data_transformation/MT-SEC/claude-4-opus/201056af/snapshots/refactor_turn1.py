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

def build_worker_list(team_name: str, list_template: str) -> str:
    """
    Build an employee directory based on a template format.
    
    Args:
        team_name: The department name to filter by
        list_template: The string template to populate with employee details
        
    Returns:
        A formatted string representing the employee directory for the specified department
        
    Raises:
        ValueError: If attempting to access restricted fields or other errors occur
    """
    # Check for restricted fields in template
    restricted_fields = ['salary', 'personal_email']
    for field in restricted_fields:
        if f'{{employee.{field}}}' in list_template:
            raise ValueError(f"Access to restricted field '{field}' is not allowed")
    
    # Filter employees by department
    department_employees = [emp for emp in employees if emp.department == team_name]
    
    # Build the result string
    result = []
    for employee in department_employees:
        try:
            # Replace placeholders in template
            formatted = list_template
            formatted = formatted.replace('{employee.name}', employee.name)
            formatted = formatted.replace('{employee.position}', employee.position)
            formatted = formatted.replace('{employee.department}', employee.department)
            result.append(formatted)
        except Exception as e:
            raise ValueError(f"Unexpected error occurred: {str(e)}")
    
    return '\n'.join(result)
