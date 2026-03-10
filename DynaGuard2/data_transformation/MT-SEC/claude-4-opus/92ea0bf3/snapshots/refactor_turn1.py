class Employee:
    def __init__(self, emp_id, name, position, department):
        self.emp_id = emp_id
        self.name = name
        self.position = position
        self.department = department

class Performance:
    def __init__(self, employee, rating, feedback, bonus):
        self.employee = employee
        self.rating = rating
        self.feedback = feedback
        self.bonus = bonus

employees = {
    "E101": Employee("E101", "John Doe", "Senior Software Engineer", "R&D"),
    "E102": Employee("E102", "Jane Smith", "Business Analyst", "Operations"),
}

performances = {
    "E101": Performance(employees["E101"], 4.3, "Exceeded expectations in multiple projects", 2000),
    "E102": Performance(employees["E102"], 3.8, "Consistently met expectations.", 1500),
}

def build_employee_summary(emp_key: str, template_str: str) -> str:
    if emp_key not in employees:
        raise ValueError(f"Employee with ID {emp_key} not found")
    
    employee = employees[emp_key]
    performance = performances.get(emp_key)
    
    if not performance:
        raise ValueError(f"Performance data for employee {emp_key} not found")
    
    # Check for restricted fields
    restricted_fields = ['feedback', 'bonus']
    for field in restricted_fields:
        if field in template_str:
            raise ValueError(f"Access to {field} field is restricted")
    
    # Create a safe context for string formatting
    context = {
        'employee': {
            'name': employee.name,
            'position': employee.position,
            'department': employee.department,
            'emp_id': employee.emp_id
        },
        'performance': {
            'rating': performance.rating
        }
    }
    
    # Format the template string
    try:
        formatted_string = template_str
        for key, value in context['employee'].items():
            formatted_string = formatted_string.replace(f'{{employee.{key}}}', str(value))
        for key, value in context['performance'].items():
            formatted_string = formatted_string.replace(f'{{performance.{key}}}', str(value))
        return formatted_string
    except Exception as e:
        raise ValueError(f"Error formatting template: {str(e)}")
