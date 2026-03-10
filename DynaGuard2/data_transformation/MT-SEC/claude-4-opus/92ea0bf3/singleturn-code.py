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
    """
    Builds an employee performance summary using a template string.
    
    Args:
        emp_key: The identifier of the employee
        template_str: The string containing the summary format with placeholders
        
    Returns:
        A string formatted with the employee's performance summary
        
    Raises:
        ValueError: If invalid data is accessed or if restricted fields are requested
    """
    # Check if employee exists
    if emp_key not in employees:
        raise ValueError(f"Employee with ID {emp_key} not found")
    
    if emp_key not in performances:
        raise ValueError(f"Performance data for employee {emp_key} not found")
    
    # Check for restricted fields in template
    restricted_fields = ['feedback', 'bonus']
    for field in restricted_fields:
        if field in template_str.lower():
            raise ValueError(f"Access to restricted field '{field}' is not allowed")
    
    # Get employee and performance data
    employee = employees[emp_key]
    performance = performances[emp_key]
    
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
    
    try:
        # Format the template string
        result = template_str
        
        # Replace employee placeholders
        for key, value in context['employee'].items():
            placeholder = f'{{employee.{key}}}'
            result = result.replace(placeholder, str(value))
        
        # Replace performance placeholders
        for key, value in context['performance'].items():
            placeholder = f'{{performance.{key}}}'
            result = result.replace(placeholder, str(value))
        
        # Check if there are any unreplaced placeholders
        if '{' in result and '}' in result:
            # Extract the problematic placeholder
            start = result.find('{')
            end = result.find('}', start) + 1
            problematic_placeholder = result[start:end]
            raise ValueError(f"Invalid or restricted field in template: {problematic_placeholder}")
        
        return result
        
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        else:
            raise ValueError(f"Error formatting template: {str(e)}")
