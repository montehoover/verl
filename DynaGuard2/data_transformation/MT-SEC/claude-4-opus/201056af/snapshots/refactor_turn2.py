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

def _validate_template(list_template: str) -> None:
    """Validate that the template doesn't contain restricted fields."""
    restricted_fields = ['salary', 'personal_email']
    for field in restricted_fields:
        if f'{{employee.{field}}}' in list_template:
            raise ValueError(f"Access to restricted field '{field}' is not allowed")

def _filter_employees_by_department(team_name: str) -> list:
    """Filter employees by department name."""
    return [emp for emp in employees if emp.department == team_name]

def _format_employee_template(employee: Employee, list_template: str) -> str:
    """Format a single employee's data using the template."""
    try:
        formatted = list_template
        formatted = formatted.replace('{employee.name}', employee.name)
        formatted = formatted.replace('{employee.position}', employee.position)
        formatted = formatted.replace('{employee.department}', employee.department)
        return formatted
    except Exception as e:
        raise ValueError(f"Unexpected error occurred: {str(e)}")

def _format_all_employees(department_employees: list, list_template: str) -> str:
    """Format all employees using the template and join them."""
    formatted_employees = []
    for employee in department_employees:
        formatted_employees.append(_format_employee_template(employee, list_template))
    return '\n'.join(formatted_employees)

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
    _validate_template(list_template)
    department_employees = _filter_employees_by_department(team_name)
    return _format_all_employees(department_employees, list_template)
