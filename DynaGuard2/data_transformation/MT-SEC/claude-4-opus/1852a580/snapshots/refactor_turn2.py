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

def validate_template_fields(template: str) -> list:
    import re
    placeholders = re.findall(r'\{employee\.(\w+)\}', template)
    restricted_fields = ['salary', 'personal_email']
    
    for field in placeholders:
        if field in restricted_fields:
            raise ValueError(f"Access to restricted field '{field}' is not allowed")
    
    return placeholders

def filter_employees_by_department(employees_list: list, department: str) -> list:
    return [emp for emp in employees_list if emp.department == department]

def format_employee_roster(employees_list: list, template: str, fields: list) -> str:
    roster_lines = []
    
    for employee in employees_list:
        line = template
        for field in fields:
            try:
                value = getattr(employee, field)
                line = line.replace(f'{{employee.{field}}}', str(value))
            except AttributeError:
                raise ValueError(f"Unknown employee attribute: {field}")
        roster_lines.append(line)
    
    return '\n'.join(roster_lines)

def generate_team_roster(division_name: str, roster_template: str) -> str:
    # Validate template and extract fields
    template_fields = validate_template_fields(roster_template)
    
    # Filter employees by department
    department_employees = filter_employees_by_department(employees, division_name)
    
    # Format and return the roster
    return format_employee_roster(department_employees, roster_template, template_fields)
