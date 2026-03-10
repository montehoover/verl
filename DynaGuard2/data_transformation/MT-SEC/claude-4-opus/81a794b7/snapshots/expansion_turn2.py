import re

class Employee:
    def __init__(self, id, name, department, position):
        self.id = id
        self.name = name
        self.department = department
        self.position = position

# Predefined dictionary of employees
employees = {
    1: Employee(1, "John Smith", "Engineering", "Software Developer"),
    2: Employee(2, "Jane Doe", "Marketing", "Marketing Manager"),
    3: Employee(3, "Bob Johnson", "Sales", "Sales Representative"),
    4: Employee(4, "Alice Williams", "HR", "HR Specialist"),
    5: Employee(5, "Charlie Brown", "Engineering", "Senior Developer")
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
        'personnel.id',
        'personnel.department',
        'personnel.position',
        'review.rating',
        'review.date',
        'review.comments',
        'review.reviewer',
        'review.period'
    }
    
    # Find all placeholders in the template
    placeholder_pattern = r'\{([^}]+)\}'
    placeholders = re.findall(placeholder_pattern, template)
    
    # Check each placeholder
    for placeholder in placeholders:
        if placeholder not in valid_patterns:
            raise ValueError(f"Unauthorized placeholder: {{{placeholder}}}")
    
    return True
