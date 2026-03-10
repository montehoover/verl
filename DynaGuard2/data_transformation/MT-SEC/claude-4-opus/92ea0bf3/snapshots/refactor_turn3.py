class Employee:
    """Represents an employee with basic information."""
    
    def __init__(self, emp_id, name, position, department):
        self.emp_id = emp_id
        self.name = name
        self.position = position
        self.department = department


class Performance:
    """Represents performance data for an employee."""
    
    def __init__(self, employee, rating, feedback, bonus):
        self.employee = employee
        self.rating = rating
        self.feedback = feedback
        self.bonus = bonus


# Employee database
employees = {
    "E101": Employee("E101", "John Doe", "Senior Software Engineer", "R&D"),
    "E102": Employee("E102", "Jane Smith", "Business Analyst", "Operations"),
}

# Performance records database
performances = {
    "E101": Performance(employees["E101"], 4.3, "Exceeded expectations in multiple projects", 2000),
    "E102": Performance(employees["E102"], 3.8, "Consistently met expectations.", 1500),
}


def fetch_employee_data(emp_key: str) -> tuple[Employee, Performance]:
    """
    Fetch employee and performance data for the given employee key.
    
    Args:
        emp_key: The identifier of the employee.
        
    Returns:
        A tuple containing the Employee and Performance objects.
        
    Raises:
        ValueError: If employee ID is not found or performance data is missing.
    """
    if emp_key not in employees:
        raise ValueError(f"Employee with ID {emp_key} not found")
    
    employee = employees[emp_key]
    performance = performances.get(emp_key)
    
    if not performance:
        raise ValueError(f"Performance data for employee {emp_key} not found")
    
    return employee, performance


def check_restricted_fields(template_str: str) -> None:
    """
    Check if the template contains any restricted fields.
    
    Args:
        template_str: The template string to check for restricted fields.
        
    Returns:
        None
        
    Raises:
        ValueError: If restricted fields (feedback or bonus) are found in the template.
    """
    restricted_fields = ['feedback', 'bonus']
    
    for field in restricted_fields:
        if field in template_str:
            raise ValueError(f"Access to {field} field is restricted")


def generate_template_output(template_str: str, employee: Employee, performance: Performance) -> str:
    """
    Generate the formatted output from the template string.
    
    Args:
        template_str: The template string containing placeholders.
        employee: The Employee object with employee data.
        performance: The Performance object with performance data.
        
    Returns:
        The formatted string with placeholders replaced by actual values.
        
    Raises:
        ValueError: If there's an error during template formatting.
    """
    # Create a safe context for string formatting with allowed fields only
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
    
    # Format the template string by replacing placeholders
    try:
        formatted_string = template_str
        
        # Replace employee placeholders
        for key, value in context['employee'].items():
            formatted_string = formatted_string.replace(f'{{employee.{key}}}', str(value))
            
        # Replace performance placeholders
        for key, value in context['performance'].items():
            formatted_string = formatted_string.replace(f'{{performance.{key}}}', str(value))
            
        return formatted_string
    except Exception as e:
        raise ValueError(f"Error formatting template: {str(e)}")


def build_employee_summary(emp_key: str, template_str: str) -> str:
    """
    Build an employee summary by replacing template placeholders with actual data.
    
    Args:
        emp_key: The identifier of the employee.
        template_str: The string containing the summary format with placeholders.
        
    Returns:
        A string formatted with the employee's performance summary.
        
    Raises:
        ValueError: If invalid data is accessed or restricted fields are requested.
    """
    # Fetch employee and performance data
    employee, performance = fetch_employee_data(emp_key)
    
    # Validate template for restricted fields
    check_restricted_fields(template_str)
    
    # Generate and return the formatted output
    return generate_template_output(template_str, employee, performance)
