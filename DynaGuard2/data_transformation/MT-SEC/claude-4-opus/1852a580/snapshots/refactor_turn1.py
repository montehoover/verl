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

def generate_team_roster(division_name: str, roster_template: str) -> str:
    import re
    
    # Find all placeholders in the template
    placeholders = re.findall(r'\{employee\.(\w+)\}', roster_template)
    
    # Check for restricted fields
    restricted_fields = ['salary', 'personal_email']
    for field in placeholders:
        if field in restricted_fields:
            raise ValueError(f"Access to restricted field '{field}' is not allowed")
    
    # Filter employees by department
    department_employees = [emp for emp in employees if emp.department == division_name]
    
    # Build the roster
    roster_lines = []
    for employee in department_employees:
        line = roster_template
        for field in placeholders:
            try:
                value = getattr(employee, field)
                line = line.replace(f'{{employee.{field}}}', str(value))
            except AttributeError:
                raise ValueError(f"Unknown employee attribute: {field}")
        roster_lines.append(line)
    
    return '\n'.join(roster_lines)
