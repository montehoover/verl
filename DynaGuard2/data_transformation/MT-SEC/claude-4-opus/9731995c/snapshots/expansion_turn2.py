import re

class Performance:
    def __init__(self, employee_id, rating, review):
        self.employee_id = employee_id
        self.rating = rating
        self.review = review

# Predefined dictionary of performance data
performance_data = {
    101: Performance(101, 4.5, "Excellent work on project deliverables"),
    102: Performance(102, 3.8, "Good teamwork and communication skills"),
    103: Performance(103, 4.2, "Strong technical skills and problem-solving"),
    104: Performance(104, 3.5, "Meets expectations, room for improvement"),
    105: Performance(105, 4.8, "Outstanding leadership and innovation")
}

def get_performance_by_id(employee_id):
    """
    Retrieves the Performance object for a given employee ID.
    
    Args:
        employee_id: The ID of the employee
        
    Returns:
        Performance object associated with the employee ID
        
    Raises:
        ValueError: If the employee ID doesn't exist in the performance data
    """
    if employee_id not in performance_data:
        raise ValueError(f"Employee ID {employee_id} not found in performance data")
    
    return performance_data[employee_id]

def check_summary_placeholders(template):
    """
    Verifies that a template string only contains valid placeholders.
    
    Args:
        template: The template string to validate
        
    Returns:
        bool: True if all placeholders are valid, False otherwise
        
    Raises:
        ValueError: If any invalid or sensitive placeholders are found
    """
    # Define valid placeholder patterns
    valid_placeholders = {
        'employee.name',
        'employee.id',
        'employee.department',
        'employee.title',
        'performance.rating',
        'performance.review',
        'performance.employee_id'
    }
    
    # Find all placeholders in the template
    placeholder_pattern = r'\{([^}]+)\}'
    found_placeholders = re.findall(placeholder_pattern, template)
    
    # Check each placeholder
    for placeholder in found_placeholders:
        # Remove any whitespace
        placeholder = placeholder.strip()
        
        # Check if it's a valid placeholder
        if placeholder not in valid_placeholders:
            # Check for potentially sensitive placeholders
            sensitive_patterns = ['password', 'ssn', 'salary', 'secret', 'private', 'confidential']
            if any(pattern in placeholder.lower() for pattern in sensitive_patterns):
                raise ValueError(f"Sensitive placeholder detected: {{{placeholder}}}")
            else:
                raise ValueError(f"Invalid placeholder: {{{placeholder}}}")
    
    return True
