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

def generate_employee_summary(employee_id: str, template: str) -> str:
    """
    Generate an employee summary based on a template string.
    
    Args:
        employee_id: The ID of the employee
        template: The summary template string with placeholders
        
    Returns:
        The generated summary string
        
    Raises:
        ValueError: If any error occurs or if access to sensitive information is attempted
    """
    try:
        # Check if employee exists
        if employee_id not in employees:
            raise ValueError(f"Employee with ID {employee_id} not found")
        
        # Check if performance data exists
        if employee_id not in performances:
            raise ValueError(f"Performance data for employee {employee_id} not found")
        
        # Check for attempts to access sensitive information
        if 'feedback' in template.lower() or 'bonus' in template.lower():
            raise ValueError("Access to sensitive information (feedback or bonus details) is not allowed")
        
        # Get employee and performance objects
        employee = employees[employee_id]
        performance = performances[employee_id]
        
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
        
        # Replace placeholders in the template
        result = template
        
        # Replace employee placeholders
        for key, value in context['employee'].items():
            placeholder = f'{{employee.{key}}}'
            result = result.replace(placeholder, str(value))
        
        # Replace performance placeholders
        for key, value in context['performance'].items():
            placeholder = f'{{performance.{key}}}'
            result = result.replace(placeholder, str(value))
        
        return result
        
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        else:
            raise ValueError(f"Error generating employee summary: {str(e)}")
