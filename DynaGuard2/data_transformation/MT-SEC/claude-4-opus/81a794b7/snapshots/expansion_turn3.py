import re

class Employee:
    def __init__(self, emp_id, name, position, department):
        self.emp_id = emp_id
        self.name = name
        self.position = position
        self.department = department

class Performance:
    def __init__(self, personnel, rating, feedback, bonus):
        self.personnel = personnel
        self.rating = rating
        self.feedback = feedback
        self.bonus = bonus

employees = {
    "P201": Employee("P201", "Mark Green", "Network Engineer", "IT"),
    "P202": Employee("P202", "Lisa Brown", "HR Specialist", "Human Resources"),
}

performances = {
    "P201": Performance(employees["P201"], 4.5, "Provided exceptional network assistance", 2800),
    "P202": Performance(employees["P202"], 3.9, "Managed complex HR cases efficiently", 2100),
}

def get_employee_by_id(employee_id):
    """
    Retrieve employee details using their ID.
    
    Args:
        employee_id: The ID of the employee to retrieve
        
    Returns:
        Employee object corresponding to the given ID
        
    Raises:
        ValueError: If the employee ID does not exist
    """
    if employee_id not in employees:
        raise ValueError(f"Employee with ID {employee_id} does not exist")
    
    return employees[employee_id]

def validate_template_placeholders(template):
    """
    Validate that all placeholders in a template string are authorized.
    
    Args:
        template: String containing placeholders like {personnel.name} or {review.rating}
        
    Returns:
        Boolean indicating if all placeholders are valid
        
    Raises:
        ValueError: If any placeholder is unauthorized or invalid
    """
    # Define valid placeholder patterns
    valid_patterns = {
        'personnel.name',
        'personnel.emp_id',
        'personnel.department',
        'personnel.position',
        'performance.rating',
        'performance.feedback',
        'performance.bonus'
    }
    
    # Find all placeholders in the template
    placeholder_pattern = r'\{([^}]+)\}'
    placeholders = re.findall(placeholder_pattern, template)
    
    # Check each placeholder
    for placeholder in placeholders:
        if placeholder not in valid_patterns:
            raise ValueError(f"Unauthorized placeholder: {{{placeholder}}}")
    
    return True

def compile_employee_evaluation(employee_id, template):
    """
    Compile a performance summary for an employee using a template.
    
    Args:
        employee_id: The ID of the employee
        template: Template string with placeholders
        
    Returns:
        Formatted summary string with placeholders replaced
        
    Raises:
        ValueError: If employee ID is invalid or unauthorized information is accessed
    """
    # Validate employee exists
    if employee_id not in employees:
        raise ValueError(f"Employee with ID {employee_id} does not exist")
    
    # Validate template placeholders
    validate_template_placeholders(template)
    
    # Get employee and performance data
    employee = employees[employee_id]
    performance = performances.get(employee_id)
    
    if not performance:
        raise ValueError(f"No performance data found for employee {employee_id}")
    
    # Create replacement dictionary
    replacements = {
        '{personnel.name}': employee.name,
        '{personnel.emp_id}': employee.emp_id,
        '{personnel.position}': employee.position,
        '{personnel.department}': employee.department,
        '{performance.rating}': str(performance.rating),
        '{performance.feedback}': performance.feedback,
        '{performance.bonus}': str(performance.bonus)
    }
    
    # Replace placeholders in template
    result = template
    for placeholder, value in replacements.items():
        result = result.replace(placeholder, value)
    
    return result
